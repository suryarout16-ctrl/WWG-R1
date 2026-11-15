#!/usr/bin/env python3
"""R1 日本語チャットボット

  PCマイク --(faster-whisper small, ja, VAD)--> テキスト
          --(OpenAI API)--> 返答
          --(PCで日本語TTS, 男性の声)--> 16kHz PCM --(AudioClient.PlayStream)--> R1のスピーカー

R1本体の TtsMaker は英語/中国語しか話せないため、音声はPCで作って PCM で流す。
既定では R1 の体は動かさない（音声だけ）。--motion を付けると、LLM が返事に合う身ぶりを選び、
話し始めと同時に R1 の Arm Action（腕のプリセット動作）を送る。立っている（Running）ときだけ送る。

例（リポジトリのルートで）:
  pixi run chat -i Ethernet                     # 本番（声で会話）
  pixi run chat --local                         # R1なしでPCのスピーカーで試す
  pixi run chat --local --text                  # マイクなし、キーボードで会話
  pixi run chat -i Ethernet --echo              # LLMなし、聞き取った文をおうむ返し
  pixi run chat -i Ethernet --say "こんにちは"   # 一言だけ話させて終了
  pixi run chat -i Ethernet --motion            # 会話に合わせて腕の身ぶり（手を振る・拍手など）も付ける
  pixi run chat --local --text --motion         # R1 なしで、LLM がどの身ぶりを選ぶかだけ確認
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

SAMPLE_RATE = 16000          # R1 の voice サービスは 16kHz / mono / s16le
CHUNK_BYTES = 32000          # 1秒ぶん。r1_audio(robonix) と同じ区切り
HERE = Path(__file__).resolve().parent
# 音声認識ワーカーを動かす Python。既定はこの pixi 環境（faster-whisper 入り）
DEFAULT_STT_PYTHON = Path(sys.executable)
# Windows で faster-whisper を GPU で動かすには cuBLAS 12 / cuDNN 9 の DLL が要る。
# そのフォルダ（例: torch 同梱の ...\site-packages\torch\lib）を環境変数 R1_CUDA_DLL_DIR に入れる。
# 無ければ GPU で失敗して自動的に CPU(int8) で動く。
DEFAULT_CUDA_DLL_DIR = os.environ.get("R1_CUDA_DLL_DIR", "")

SYSTEM_PROMPT = """あなたはUnitree社のヒューマノイドロボット「R1」です。ハッカソン会場で来場者と話しています。
- 一人称は「僕」。落ち着いた、親しみやすい男性の話し言葉で答える
- 返答は日本語で1〜2文、60文字程度まで。音声で読み上げるので短く
- 記号、絵文字、箇条書き、URL、英字の略語は使わない。数字は読みやすく"""

NO_MOTION_RULE = "\n- 体を動かす約束はしない（今は話すことだけができる）"

MOTION_RULE = """
- 返事と一緒に、腕の身ぶりを gesture から1つ選べる。say には読み上げる日本語だけを書く
  wave: あいさつ・お別れ / wave_big: 大きく手を振る（遠くの人へ、盛り上げたいとき）/ clap: 拍手・お祝い
  fist_pump: やったね・がんばろう / hand_on_heart: 感謝や気持ちを伝える / emphasize: 説明で強調したいとき
- 合う身ぶりが無ければ none。毎回は動かさない（3回に1回くらい）。「手を振って」など頼まれた身ぶりが上にあれば選ぶ
- 会話中は歩く・踊る・物を持つ・握手・ハグはしない。頼まれたら、できる身ぶりを提案して話で返す"""

# 会話から呼べる腕の動作（R1 Arm Action のプリセット ID）。人に触れる動作（握手・ハグ・ハイタッチ）は入れない
GESTURES = {
    "wave": (25, "胸の前で手を振る"),
    "wave_big": (26, "頭の上で手を振る"),
    "clap": (17, "拍手"),
    "fist_pump": (29, "ガッツポーズ（右）"),
    "hand_on_heart": (33, "右手を胸に当てる"),
    "emphasize": (35, "強調のしぐさ"),
    "point_forward": (31, "前方を示す"),  # 既存プリセット。対象や指先の位置は自動調整しない。
    "hat_tip": (None, "帽子をかぶる身ぶり（教示動作）"),
}
HOLD_IDS = {29, 31, 33, 35}   # 終わったあと姿勢を保つ動作 → 話し終わったら 99 で腕を戻す
STANDING_FSM = {811, 816}  # Running（816 は Arm Action 実行中に観測）


REPO_ROOT = HERE.parent
_log_file = None       # 画面と同じ内容を時刻付きで書く .log
_turn_file = None      # 1往復ごとの記録（.jsonl）


def setup_logging(log_dir: str, mode: str) -> Path:
    global _log_file, _turn_file
    stamp = time.strftime("%Y%m%d_%H%M%S")
    folder = Path(log_dir)
    folder.mkdir(parents=True, exist_ok=True)
    base = folder / f"r1_chat_{stamp}_{mode}"
    _log_file = open(f"{base}.log", "a", encoding="utf-8", buffering=1)
    _turn_file = open(f"{base}_turns.jsonl", "a", encoding="utf-8", buffering=1)
    return base


def log(message: str) -> None:
    print(message, flush=True)
    if _log_file:
        for line in str(message).splitlines() or [""]:
            _log_file.write(f"{time.strftime('%H:%M:%S')} {line}\n")


def log_turn(**record) -> None:
    if _turn_file:
        _turn_file.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- audio utils
def to_pcm16k(samples: np.ndarray, rate: int) -> bytes:
    """mono の int16/float 配列を 16kHz s16le bytes にする。"""
    x = samples.astype(np.float32)
    if samples.dtype == np.int16:
        x /= 32768.0
    if rate != SAMPLE_RATE:
        g = np.gcd(rate, SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, rate // g)
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype("<i2").tobytes()


def wav_to_pcm16k(data: bytes) -> bytes:
    with wave.open(io.BytesIO(data)) as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"16bit WAV 以外は未対応です (sampwidth={w.getsampwidth()})")
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        if w.getnchannels() > 1:
            frames = frames.reshape(-1, w.getnchannels()).mean(axis=1).astype(np.int16)
        return to_pcm16k(frames, w.getframerate())


def boost(pcm: bytes, gain_db: float) -> bytes:
    """R1 の音量100より大きく聞かせるため、PCM を持ち上げてソフトに頭打ちさせる（音割れを抑えた音圧上げ）。"""
    if gain_db <= 0:
        return pcm
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    g = 10 ** (gain_db / 20)
    y = np.tanh(x * g) / np.tanh(g)      # 小さい音は約 g 倍、大きい音は 1.0 手前で丸める
    return (np.clip(y, -1.0, 1.0) * 32000).astype("<i2").tobytes()


def clean_for_speech(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_#`>|~\-]{1,}", " ", text)
    text = re.sub(r"[\U0001F000-\U0001FAFF☀-➿]", "", text)  # 絵文字
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- TTS (PC側)
# PowerShell を1つ起動したままにして、1行 "入力txt<TAB>出力wav" ごとに合成する（毎回起動するより約2秒速い）
SAPI_SCRIPT = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try { $s.SelectVoice($env:R1_TTS_VOICE) } catch { [Console]::Out.WriteLine("novoice " + $s.Voice.Name); [Console]::Out.Flush(); exit 3 }
$s.Rate = [int]$env:R1_TTS_RATE
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
[Console]::Out.WriteLine("ready " + $s.Voice.Name); [Console]::Out.Flush()
while ($null -ne ($line = [Console]::In.ReadLine())) {
  try {
    $p = $line -split "`t"
    $s.SetOutputToWaveFile($p[1], $fmt)
    $s.Speak([IO.File]::ReadAllText($p[0], [Text.Encoding]::UTF8))
    $s.SetOutputToNull()
    [Console]::Out.WriteLine("ok")
  } catch { [Console]::Out.WriteLine("error " + $_.Exception.Message) }
  [Console]::Out.Flush()
}
"""


class SapiTTS:
    """Windows標準の日本語音声。オフラインで動く。男性は Microsoft Ichiro。"""

    # Ichiro・David・Mark などの新しい音声（OneCore）は PowerShell 7 の System.Speech からしか選べない。
    # Windows PowerShell 5.1 では選択に失敗し、既定の女性の声（Haruka）で黙って話してしまうため、
    # pwsh を先に使い、どちらでも選べなければ起動を止める（別の声で話さない）。
    SHELLS = ("pwsh", "powershell.exe")

    def __init__(self, voice: str, rate: int):
        env = dict(os.environ, R1_TTS_VOICE=voice, R1_TTS_RATE=str(rate))
        self.proc, self.voice_name, tried = None, "", []
        for shell in self.SHELLS:
            exe = shutil.which(shell)
            if not exe:
                continue
            proc = subprocess.Popen([exe, "-NoProfile", "-NonInteractive", "-Command", SAPI_SCRIPT],
                                    env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, encoding="utf-8", errors="replace")
            first = proc.stdout.readline().strip()
            if first.startswith("ready"):
                self.proc, self.voice_name = proc, first[len("ready"):].strip()
                break
            proc.kill()
            tried.append(f"{shell}: {first or proc.stderr.read()[:200]}")
        if self.proc is None:
            raise RuntimeError(f"Windows音声 '{voice}' を選べません（{' / '.join(tried) or 'PowerShell が見つかりません'}）")
        self.tmp = Path(tempfile.mkdtemp(prefix="r1_tts_"))

    def synth(self, text: str) -> bytes:
        src, out = self.tmp / "in.txt", self.tmp / "out.wav"
        src.write_text(text, encoding="utf-8")
        self.proc.stdin.write(f"{src}\t{out}\n")
        self.proc.stdin.flush()
        status = self.proc.stdout.readline().strip()
        if status != "ok":
            raise RuntimeError(f"Windows音声の合成に失敗しました: {status}")
        return wav_to_pcm16k(out.read_bytes())


class OpenAITTS:
    def __init__(self, client, voice: str, model: str):
        self.client, self.voice, self.model = client, voice, model

    def synth(self, text: str) -> bytes:
        kwargs = dict(model=self.model, voice=self.voice, input=text, response_format="pcm")
        if "gpt-4o" in self.model:
            kwargs["instructions"] = "落ち着いた成人男性の声で、自然な日本語のイントネーションで話してください。"
        audio = self.client.audio.speech.create(**kwargs).read()   # 24kHz s16le mono
        return to_pcm16k(np.frombuffer(audio, dtype="<i2"), 24000)


class Sbv2TTS:
    """Style-Bert-VITS2 の API サーバ (server_fastapi.py) に問い合わせる。男性モデルを指定して使う。"""

    def __init__(self, url: str, model_name: str, style: str):
        self.url, self.model_name, self.style = url.rstrip("/"), model_name, style

    def synth(self, text: str) -> bytes:
        import urllib.parse
        import urllib.request
        query = {"text": text, "language": "JP", "style": self.style}
        if self.model_name:
            query["model_name"] = self.model_name
        with urllib.request.urlopen(f"{self.url}/voice?{urllib.parse.urlencode(query)}", timeout=60) as r:
            return wav_to_pcm16k(r.read())


# ---------------------------------------------------------------- 再生先
class RobotSpeaker:
    APP_NAME = "r1_chat"

    def __init__(self, interface: str, volume: int):
        if not interface.isascii():
            sys.exit(f"-i に日本語の名前 ({interface}) は使えません。アダプタ名を Ethernet などに変えてください（資料01 Step6）")
        Path("/tmp").mkdir(exist_ok=True)   # Windows で CycloneDDS のログ置き場
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient   # R1専用の音声クライアントは無いので G1 用を使う（資料03と同じ）
        ChannelFactoryInitialize(0, interface)
        self.client = AudioClient()
        self.client.SetTimeout(10.0)
        self.client.Init()
        time.sleep(1.0)   # DDS の相手探し
        code = self.client.SetVolume(volume)
        if code != 0:
            sys.exit(f"R1 の音声サービスに届きません (SetVolume code={code})。02 の ping と sample-tts を先に確認してください")
        log(f"[R1] 接続OK  音量={volume}")

    def play(self, pcm: bytes, *, should_continue=None) -> None:
        if should_continue is not None and not should_continue():
            return
        stream_id = uuid.uuid4().hex
        sent = False
        try:
            for i in range(0, len(pcm), CHUNK_BYTES):
                if should_continue is not None and not should_continue():
                    return
                sent = True   # RPC 例外でも、送信を試みたストリームは停止する。
                code, _ = self.client.PlayStream(self.APP_NAME, stream_id, pcm[i:i + CHUNK_BYTES])
                if code != 0:
                    raise RuntimeError(f"PlayStream 失敗 code={code}")
            wait_s = len(pcm) / (SAMPLE_RATE * 2) + 0.3
            if should_continue is None:
                time.sleep(wait_s)
            else:
                deadline = time.monotonic() + wait_s
                while should_continue():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    time.sleep(min(0.05, remaining))
        finally:
            if sent or should_continue is None:
                self.client.PlayStop(self.APP_NAME)


class LocalSpeaker:
    def play(self, pcm: bytes, *, should_continue=None) -> None:
        if should_continue is not None and not should_continue():
            return
        import sounddevice as sd
        sd.play(np.frombuffer(pcm, dtype="<i2"), SAMPLE_RATE)
        if should_continue is None:
            sd.wait()
            return
        stream = sd.get_stream()
        while stream.active:
            if not should_continue():
                sd.stop()
                return
            time.sleep(0.05)


# ---------------------------------------------------------------- 腕の身ぶり (R1 Arm Action)
class Gesturer:
    """プリセット／登録済み教示動作を送る。lock は会話と手動操作の共用ロック。"""

    FINISH_TIMEOUT = 12.0
    MAX_CUSTOM_DURATION = 10.0

    def __init__(self, dry: bool = False, hat_action: str = "hiyari_hat_v1", hat_fallback: bool = False):
        self.dry = dry
        self.hat_action = hat_action
        self.hat_fallback = hat_fallback
        self.lock = threading.RLock()
        self.thread = None
        self.action_id = None
        self.custom_name = None
        self.custom_duration = 0.0
        self.result: dict = {}
        if dry:
            self.describe = lambda code: "OK" if code == 0 else f"code={code}"
            log(f"[動作] --local のため送信しません。帽子の教示動作: {hat_action}")
            return
        sys.path.insert(0, str(REPO_ROOT / "r1_motion"))
        from arm_action import ACTIONS, connect, describe   # ChannelFactoryInitialize の後で呼ぶこと
        from stand import R1Loco
        self.describe = describe
        self.allowed_action_ids = set(ACTIONS)
        self.arm = connect(self.FINISH_TIMEOUT)
        self.loco = R1Loco()
        self.loco.SetTimeout(3.0)
        self.loco.Init()
        log(f"[動作] 身ぶりを使います: {', '.join(f'{k}={v[1]}' for k, v in GESTURES.items())}")

    def _ready(self) -> str | None:
        try:
            fsm = self.loco.get_fsm()
        except Exception as error:
            self.result.update(code=7399, error=f"{type(error).__name__}: {error}")
            log(f"  [動作] モードを読めないので送らない: {error}")
            return "fsm_error"
        if fsm not in STANDING_FSM:
            self.result.update(code=7404, fsm=fsm)
            log(f"  [動作] 立っていない（モード {fsm}）ので送らない")
            return f"skip_fsm_{fsm}"
        return None

    def _preset(self, action_id: int) -> int:
        code = self.arm.execute(action_id)
        if code == 7401 and action_id != 99:
            release = self.arm.release()
            if release != 0:
                return release
            code = self.arm.execute(action_id)
        return code

    def start(self, name: str) -> str:
        with self.lock:
            # 完了済みでも finish が保持姿勢を解除するまでは次の命令を送らない。
            if self.thread is not None:
                log("  [動作] 前の身ぶりの終了処理がまだなので送らない")
                return "busy"
            if name not in GESTURES:
                self.result = {"status": "none", "code": 3204}
                return "none"
            action_id, label = GESTURES[name]
            self.result = {"gesture": name, "label": label}
            self.action_id, self.custom_name, self.custom_duration = action_id, None, 0.0
            if self.dry:
                choice = f"教示名 {self.hat_action}" if name == "hat_tip" else f"ID {action_id}"
                self.result.update(status="dry", code=0, action_name=self.hat_action if name == "hat_tip" else None)
                log(f"  [動作] {label}（{choice}、--local なので送らない）")
                return "dry"
            status = self._ready()
            if status:
                self.result["status"] = status
                return status
            status = "sent"
            if name == "hat_tip":
                try:
                    code, actions = self.arm.action_list()
                except Exception as error:
                    self.result.update(status="list_error", code=7399, error=f"{type(error).__name__}: {error}")
                    log(f"  [動作] 教示一覧を読めないので送らない: {error}")
                    return "list_error"
                if code != 0 or not isinstance(actions, list) or len(actions) < 2 or not isinstance(actions[1], list):
                    self.result.update(status="list_error", code=code if code else 7399)
                    log("  [動作] 教示一覧が確認できないので帽子の動作は送らない")
                    return "list_error"
                entry = next((item for item in actions[1] if isinstance(item, dict) and item.get("name") == self.hat_action), None)
                if entry is None:
                    if not self.hat_fallback:
                        self.result.update(status="unavailable_hat_action", code=7402, action_name=self.hat_action)
                        log(f"  [動作] 教示動作 {self.hat_action} が未登録なので送らない（帽子の動作は未確認）")
                        return "unavailable_hat_action"
                    self.action_id = 26
                    label = "頭の上で手を振る（帽子の身ぶりの代替）"
                    self.result.update(label=label, fallback=True)
                    status = "sent_fallback"
                else:
                    duration = entry.get("time")
                    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 < duration <= self.MAX_CUSTOM_DURATION:
                        self.result.update(status="invalid_hat_duration", code=3204, action_name=self.hat_action)
                        log(f"  [動作] 教示動作 {self.hat_action} の長さが未確認／上限 {self.MAX_CUSTOM_DURATION:g} 秒超なので送らない")
                        return "invalid_hat_duration"
                    self.custom_name, self.custom_duration = self.hat_action, float(duration)
                    self.result.update(action_name=self.custom_name, duration=self.custom_duration)
                    status = "sent_custom"
            self.result.update(status=status, action_id=self.action_id)

            def run():
                try:
                    code = self.arm.execute_custom(self.custom_name) if self.custom_name else self._preset(self.action_id)
                    self.result["code"] = code
                    # 7108 は非同期。登録時間が経過するまで同じ動作を占有する。
                    if code == 0 and self.custom_name:
                        time.sleep(self.custom_duration)
                except Exception as error:
                    self.result.update(code=7399, error=f"{type(error).__name__}: {error}")
                    log(f"  [動作] 実行エラー: {error}")

            self.thread = threading.Thread(target=run, name="r1-gesture", daemon=False)
            self.thread.start()
            choice = f"教示名 {self.custom_name}" if self.custom_name else f"ID {self.action_id}"
            log(f"  [動作] {label}（{choice}）")
            return status

    def manual(self, action_id: int) -> int:
        """手動プリセット。99 は終了済み動作の後処理も再試行できる。"""
        with self.lock:
            if isinstance(action_id, bool) or not isinstance(action_id, int):
                return 3204
            if self.thread is not None:
                if action_id != 99 or self.thread.is_alive():
                    return 7400
                cleanup = self.finish()
                if self.thread is not None:
                    return cleanup if cleanup is not None else 7399
                if cleanup == 0:
                    # finish が保持姿勢／教示再生を既に解除したので重複送信しない。
                    log("  [動作] 保留されていた動作の終了処理を完了しました")
                    return 0
                # 実行エラーを記録して終了済みなら、明示された 99 で復帰を試す。
            if self.dry:
                log(f"  [動作] 手動 ID {action_id}（--local なので送らない）")
                return 0
            if action_id not in self.allowed_action_ids:
                return 7402
            self.result = {"status": "manual", "action_id": action_id}
            if action_id != 99 and self._ready():
                return self.result.get("code", 7404)
            try:
                code = self._preset(action_id)
                self.result["code"] = code
                return code
            except Exception as error:
                self.result.update(code=7399, error=f"{type(error).__name__}: {error}")
                log(f"  [動作] 手動操作エラー: {error}")
                return 7399

    def perform(self, name: str) -> int:
        """手動の名前付き動作を開始から終了処理まで直列実行する。"""
        with self.lock:
            status = self.start(name)
            if status == "busy":
                return 7400
            if status == "dry":
                return 0
            if not status.startswith("sent"):
                return self.result.get("code", 7399)
            return self.finish()

    def finish(self):
        """動作の終了まで待つ。タイムアウト中は参照を保持し、重複送信を防ぐ。"""
        with self.lock:
            if self.dry or not self.thread:
                return None
            self.thread.join(timeout=self.FINISH_TIMEOUT + self.custom_duration)
            if self.thread.is_alive():
                self.result["finish_status"] = "busy_timeout"
                log("  [動作] 実行待ちが続いています。終了を確認するまで次の動作は送れません")
                return 3104
            code = self.result.get("code", 7399)
            if code != 0:
                log(f"  [動作] 結果: {self.describe(code)}")
            if code == 0 and (self.custom_name or self.action_id in HOLD_IDS):
                try:
                    release = self.arm.stop_custom() if self.custom_name else self.arm.release()
                except Exception as error:
                    self.result.update(finish_status="release_error", error=f"{type(error).__name__}: {error}")
                    log(f"  [動作] 腕を戻す操作でエラー: {error}")
                    return 7399
                self.result["release_code"] = release
                if release != 0:
                    self.result["finish_status"] = "release_error"
                    log(f"  [動作] 腕を戻せません: {self.describe(release)}")
                    return release
            self.result["finish_status"] = "done"
            self.thread = None
            return code


# ---------------------------------------------------------------- 音声認識 (faster-whisper)
_KANA_KANJI = re.compile(r"[぀-ヿ㐀-鿿ｦ-ﾟ]")
_LATIN = re.compile(r"[A-Za-z]")


def text_language(text: str) -> str:
    """文字から返答言語を決める。かな・漢字が 1 文字でもあれば日本語、ほぼ英字だけなら英語。

    「A」「start」など英字が混ざる日本語は日本語のまま。判定できないときは日本語。
    """
    if not isinstance(text, str) or not text.strip():
        return "ja"
    if _KANA_KANJI.search(text):
        return "ja"
    letters = len(_LATIN.findall(text))
    return "en" if letters >= 2 and letters >= 0.5 * len(re.sub(r"[\s\d\W_]", "", text) or "x") else "ja"


class WhisperSTT:
    def __init__(self, python: Path, model: str, device: str, compute_type: str, dll_dir: str = "",
                 language: str = "ja"):
        if not Path(python).exists():
            sys.exit(f"faster-whisper の Python が見つかりません: {python}  (--stt-python で指定)")
        self.language = language
        log(f"[STT] faster-whisper ({model}, {device}, {compute_type}, 言語={language}) を読み込み中… 初回はモデルのダウンロードで時間がかかります")
        extra = ["--dll-dir", dll_dir] if dll_dir else []
        self.proc = subprocess.Popen(
            [str(python), str(HERE / "stt_worker.py"), "--model", model, "--device", device,
             "--compute-type", compute_type, "--language", language, "--vad-filter", *extra],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
            env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        ready = self._read()
        if ready.get("type") != "ready":
            sys.exit(f"[STT] 起動に失敗しました: {ready}")
        log(f"[STT] 準備OK ({ready.get('device')})")

    def _read(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("STT ワーカーが終了しました（上のエラーを確認してください）")
        return json.loads(line)

    def transcribe_detail(self, wav_path: str) -> dict:
        """{"text", "language", "language_prob", "confidence", "avg_logprob"} を返す。"""
        rid = uuid.uuid4().hex
        self.proc.stdin.write(json.dumps({"id": rid, "audioPath": str(wav_path)}) + "\n")
        self.proc.stdin.flush()
        msg = self._read()
        if msg.get("type") == "error":
            raise RuntimeError(f"音声認識エラー: {msg.get('message')}")
        return {"text": msg.get("text", ""), "language": msg.get("language") or self.language.split(",")[0],
                "language_prob": msg.get("language_prob"), "confidence": msg.get("confidence"),
                "avg_logprob": msg.get("avg_logprob"), "retried_as": msg.get("retried_as")}

    def transcribe(self, wav_path: str) -> str:
        return self.transcribe_detail(wav_path)["text"]

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.terminate()


def record_push_to_talk(device) -> np.ndarray:
    import sounddevice as sd
    frames: list[np.ndarray] = []
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", device=device,
                            callback=lambda data, *_: frames.append(data.copy()))
    with stream:
        input("  🎙 録音中… 話し終わったら Enter ")
    return np.concatenate(frames)[:, 0] if frames else np.zeros(0, dtype=np.int16)


# ---------------------------------------------------------------- LLM
class Brain:
    def __init__(self, client, model: str, effort: str, motion: bool = False):
        self.client, self.model, self.effort, self.motion = client, model, effort, motion
        self.previous_id = None
        self.instructions = SYSTEM_PROMPT + (MOTION_RULE if motion else NO_MOTION_RULE)

    def reply(self, user_text: str) -> tuple[str, str]:
        """(読み上げる文, 身ぶり名 or "none") を返す。"""
        kwargs = dict(model=self.model, instructions=self.instructions, input=user_text,
                      previous_response_id=self.previous_id, max_output_tokens=400)
        if self.motion:   # 文と身ぶりを JSON で受け取る（Structured Outputs）
            kwargs["text"] = {"format": {
                "type": "json_schema", "name": "r1_reply", "strict": True,
                "schema": {"type": "object", "additionalProperties": False, "required": ["say", "gesture"],
                           "properties": {"say": {"type": "string"},
                                          "gesture": {"type": "string", "enum": ["none", *GESTURES]}}}}}
        if self.effort:
            kwargs["reasoning"] = {"effort": self.effort}
        try:
            resp = self.client.responses.create(**kwargs)
        except Exception as error:   # reasoning 非対応モデルなら外してやり直す
            text = str(error).lower()
            if self.effort and "reasoning" in text and ("unsupported" in text or "not supported" in text):
                log(f"[LLM] {self.model} は reasoning 指定に非対応のため外します: {str(error)[:200]}")
                self.effort = ""
                return self.reply(user_text)
            raise
        out = resp.output_text.strip()
        if not out:   # 推論でトークンを使い切った等。会話の流れに残さず、呼び出し側で定型文にする
            details = getattr(resp, "incomplete_details", None)
            raise RuntimeError(f"返答が空でした (status={resp.status}, {details})")
        self.previous_id = resp.id
        if not self.motion:
            return out, "none"
        try:
            data = json.loads(out)
            return str(data.get("say", "")).strip(), str(data.get("gesture", "none"))
        except (json.JSONDecodeError, AttributeError):
            return out, "none"


# ---------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description="R1 日本語チャットボット")
    p.add_argument("-i", "--interface", help="R1 につないだ有線LANアダプタ名（例: Ethernet）")
    p.add_argument("--local", action="store_true", help="R1 を使わず PC のスピーカーで鳴らす")
    p.add_argument("--text", action="store_true", help="マイクを使わずキーボードで会話する")
    p.add_argument("--say", help="この文だけ話して終了（LLM・マイクは使わない）")
    p.add_argument("--echo", action="store_true", help="LLM を使わず、聞き取った文をそのまま話す（接続確認用）")
    p.add_argument("--volume", type=int, default=70)
    p.add_argument("--gain", type=float, default=0.0, help="音声をさらに持ち上げる量(dB)。6〜12 くらい。上げすぎると歪む")
    p.add_argument("--tts", choices=("sapi", "openai", "sbv2"), default="sapi")
    p.add_argument("--voice", help="sapi: 'Microsoft Ichiro'(既定) / openai: onyx(既定) など")
    p.add_argument("--rate", type=int, default=0, help="sapi の話す速さ -10〜10")
    p.add_argument("--openai-tts-model", default="gpt-4o-mini-tts")
    p.add_argument("--sbv2-url", default="http://127.0.0.1:5000")
    p.add_argument("--sbv2-model", default="jvnv-M1-jp")
    p.add_argument("--sbv2-style", default="Neutral")
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5-mini"), help="OpenAI のモデル名")
    p.add_argument("--reasoning-effort", default="minimal", help="推論の深さ。空文字で指定しない")
    p.add_argument("--stt-python", default=os.environ.get("R1_STT_PYTHON", str(DEFAULT_STT_PYTHON)))
    p.add_argument("--stt-model", default="small")
    p.add_argument("--stt-device", default="cuda")
    p.add_argument("--stt-compute-type", default="float16")
    p.add_argument("--cuda-dll-dir", default=DEFAULT_CUDA_DLL_DIR)
    p.add_argument("--mic-device", default=None, help="sounddevice の入力デバイス番号か名前")
    p.add_argument("--log-dir", default=str(REPO_ROOT / "logs"), help="ログの保存先")
    p.add_argument("--no-log", action="store_true", help="ログを残さない")
    p.add_argument("--motion", action="store_true",
                   help="会話に合わせて腕の身ぶりを付ける（R1 が Running のときだけ送る。--local では表示だけ）")
    p.add_argument("--save-audio", action="store_true", help="録音した声も logs/audio に WAV で残す（認識の確認用）")
    args = p.parse_args()

    if not args.local and not args.interface:
        p.error("-i <有線LANアダプタ名> か --local のどちらかを指定してください")

    if not args.no_log:
        mode = "say" if args.say is not None else "echo" if args.echo else "chat"
        base = setup_logging(args.log_dir, mode)
        log(f"[log] {base}.log / {base.name}_turns.jsonl")
        log(f"[設定] " + " ".join(f"{k}={v}" for k, v in vars(args).items() if k not in ("say",)))
    try:
        run(args)
    except SystemExit as error:
        if error.code not in (None, 0):
            log(f"[終了] {error.code}")
        raise
    except Exception as error:
        import traceback
        log(f"[エラー] {type(error).__name__}: {error}\n{traceback.format_exc()}")
        raise


def run(args) -> None:

    needs_openai = (args.say is None and not args.echo) or args.tts == "openai"
    client = None
    if needs_openai:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:   # 無ければ Windows 資格情報マネージャー（このPCの中だけに保存）から読む
            from credentials import get_secret
            api_key = get_secret()
        if not api_key:
            sys.exit("OpenAI の API キーがありません。`pixi run python r1_chat/credentials.py set` で保存してください")
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

    if args.tts == "sapi":
        tts = SapiTTS(args.voice or "Microsoft Ichiro", args.rate)
    elif args.tts == "openai":
        tts = OpenAITTS(client, args.voice or "onyx", args.openai_tts_model)
    else:
        tts = Sbv2TTS(args.sbv2_url, args.sbv2_model, args.sbv2_style)

    speaker = LocalSpeaker() if args.local else RobotSpeaker(args.interface, args.volume)

    use_motion = args.motion and args.say is None and not args.echo
    gesturer = Gesturer(dry=args.local) if use_motion else None

    def speak(text: str, gesture: str = "none", record: dict | None = None) -> float:
        """話して、音声合成にかかった秒数を返す。gesture があれば話し始めと同時に腕を動かす。"""
        text = clean_for_speech(text)
        if not text:
            return 0.0
        t0 = time.time()
        pcm = boost(tts.synth(text), args.gain)
        synth_s = time.time() - t0
        log(f"  (音声合成 {synth_s:.1f}s, {len(pcm) / (SAMPLE_RATE * 2):.1f}s ぶん)")
        status = gesturer.start(gesture) if gesturer and gesture != "none" else None
        try:
            speaker.play(pcm)
        finally:
            if status and status.startswith("sent"):
                code = gesturer.finish()
                if record is not None:
                    record["gesture_code"] = code
        if record is not None and status:
            record["gesture_status"] = status
        return synth_s

    if args.say is not None:
        log(f"R1 > {args.say}")
        record = {"mode": "say", "answer": args.say}
        try:
            record["tts_s"] = round(speak(args.say), 2)
            log("  発話OK")
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            log_turn(**record)
        return

    brain = None if args.echo else Brain(client, args.model, args.reasoning_effort, motion=use_motion)
    stt = None if args.text else WhisperSTT(args.stt_python, args.stt_model, args.stt_device,
                                            args.stt_compute_type, args.cuda_dll_dir)
    mic = int(args.mic_device) if args.mic_device and args.mic_device.isdigit() else args.mic_device

    log("\n=== R1 チャット開始 ===")
    log("文字を打って Enter → そのまま送信 / 何も打たずに Enter → 声で話す / q → 終了" if stt
        else "文字を打って Enter で送信 / q で終了")
    try:
        while True:
            typed = input("\nあなた > ").strip()
            if typed.lower() in ("q", "quit", "exit"):
                break
            record = {"mode": "echo" if brain is None else "chat"}
            if typed:
                user_text = typed
                record["input"] = "text"
                if _log_file:
                    _log_file.write(f"{time.strftime('%H:%M:%S')} あなた > {typed}\n")
            elif stt:
                audio = record_push_to_talk(mic)
                record.update(input="voice", audio_s=round(len(audio) / SAMPLE_RATE, 2))
                if len(audio) < SAMPLE_RATE * 0.3:
                    log("  (短すぎるので無視しました)")
                    continue
                if args.save_audio and not args.no_log:
                    audio_dir = Path(args.log_dir) / "audio"
                    audio_dir.mkdir(parents=True, exist_ok=True)
                    wav_path = str(audio_dir / f"{time.strftime('%Y%m%d_%H%M%S')}.wav")
                    record["audio_file"] = wav_path
                else:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        wav_path = f.name
                with wave.open(wav_path, "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
                    w.writeframes(audio.astype("<i2").tobytes())
                t0 = time.time()
                try:
                    user_text = stt.transcribe(wav_path)
                except Exception as error:
                    log(f"  [認識エラー] {type(error).__name__}: {error}")
                    log_turn(**record, error=f"stt: {error}")
                    continue
                finally:
                    if "audio_file" not in record:
                        os.unlink(wav_path)
                record["stt_s"] = round(time.time() - t0, 2)
                log(f"  認識 ({record['stt_s']:.1f}s): {user_text or '（聞き取れませんでした）'}")
                if not user_text:
                    log_turn(**record, user="")
                    continue
            else:
                continue
            record["user"] = user_text

            t0 = time.time()
            gesture = "none"
            if brain is None:   # --echo: 聞き取った文をそのまま返す（キーなしで経路だけ確認）
                answer = f"今、{user_text}、と聞こえました。"
            else:
                try:
                    answer, gesture = brain.reply(user_text)
                except Exception as error:   # 回線切れなどで止まらないようにする
                    log(f"  [LLM エラー] {type(error).__name__}: {str(error)[:200]}")
                    record["error"] = f"llm: {type(error).__name__}: {str(error)[:200]}"
                    answer = "ごめんなさい、今ちょっと通信がうまくいきません。もう一度話しかけてください。"
            record.update(answer=answer, llm_s=round(time.time() - t0, 2))
            if use_motion:
                record["gesture"] = gesture
            shown = f"  ［身ぶり: {GESTURES[gesture][1]}］" if gesture in GESTURES else ""
            log(f"R1 ({record['llm_s']:.1f}s) > {answer}{shown}")
            try:
                record["tts_s"] = round(speak(answer, gesture, record), 2)
            except Exception as error:
                log(f"  [発話エラー] {type(error).__name__}: {error}")
                record["error"] = f"speak: {type(error).__name__}: {error}"
            log_turn(**record)
    except (KeyboardInterrupt, EOFError):
        log("\n終了します")
    finally:
        if stt:
            stt.close()


if __name__ == "__main__":
    main()
