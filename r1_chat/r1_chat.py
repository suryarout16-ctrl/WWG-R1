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
import os
import re
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
}
HOLD_IDS = {29, 33, 35}   # 終わったあと姿勢を保つ動作 → 話し終わったら 99 で腕を戻す
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
$s.SelectVoice($env:R1_TTS_VOICE)
$s.Rate = [int]$env:R1_TTS_RATE
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
[Console]::Out.WriteLine("ready"); [Console]::Out.Flush()
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

    def __init__(self, voice: str, rate: int):
        env = dict(os.environ, R1_TTS_VOICE=voice, R1_TTS_RATE=str(rate))
        self.proc = subprocess.Popen(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", SAPI_SCRIPT],
                                     env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, encoding="utf-8", errors="replace")
        if self.proc.stdout.readline().strip() != "ready":
            raise RuntimeError(f"Windows音声 '{voice}' を使えません: {self.proc.stderr.read()[:300]}")
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

    def play(self, pcm: bytes) -> None:
        stream_id = uuid.uuid4().hex
        try:
            for i in range(0, len(pcm), CHUNK_BYTES):
                code, _ = self.client.PlayStream(self.APP_NAME, stream_id, pcm[i:i + CHUNK_BYTES])
                if code != 0:
                    raise RuntimeError(f"PlayStream 失敗 code={code}")
            time.sleep(len(pcm) / (SAMPLE_RATE * 2) + 0.3)   # 送ったぶん鳴り終わるのを待つ
        finally:
            self.client.PlayStop(self.APP_NAME)


class LocalSpeaker:
    def play(self, pcm: bytes) -> None:
        import sounddevice as sd
        sd.play(np.frombuffer(pcm, dtype="<i2"), SAMPLE_RATE)
        sd.wait()


# ---------------------------------------------------------------- 腕の身ぶり (R1 Arm Action)
class Gesturer:
    """LLM が選んだ身ぶりを、話し始めと同時に送る。dry=True（--local）なら送らずに表示だけ。"""

    def __init__(self, dry: bool):
        self.dry = dry
        self.thread = None
        self.action_id = None
        self.result: dict = {}
        if dry:
            log("[動作] --local のため身ぶりは送らず、選ばれた身ぶりを表示だけします")
            return
        sys.path.insert(0, str(REPO_ROOT / "r1_motion"))
        from arm_action import connect, describe   # ChannelFactoryInitialize の後で呼ぶこと
        from stand import R1Loco
        self.describe = describe
        self.arm = connect(12.0)
        self.loco = R1Loco()
        self.loco.SetTimeout(3.0)
        self.loco.Init()
        log(f"[動作] 身ぶりを使います: {', '.join(f'{k}={v[1]}' for k, v in GESTURES.items())}")

    def start(self, name: str) -> str:
        if name not in GESTURES:
            return "none"
        action_id, label = GESTURES[name]
        if self.dry:
            log(f"  [動作] {label}（ID {action_id}、--local なので送らない）")
            return "dry"
        if self.thread and self.thread.is_alive():
            log("  [動作] 前の身ぶりがまだ動いているので送らない")
            return "busy"
        try:
            fsm = self.loco.get_fsm()
        except Exception as error:
            log(f"  [動作] モードを読めないので送らない: {error}")
            return "fsm_error"
        if fsm not in STANDING_FSM:
            log(f"  [動作] 立っていない（モード {fsm}）ので送らない")
            return f"skip_fsm_{fsm}"
        self.action_id, self.result = action_id, {}

        def run():
            code = self.arm.execute(action_id)   # 動作が終わるまで戻らない
            if code == 7401:   # 前の動作（握手など）で腕が止まったまま → 戻してから送り直す
                self.arm.release()
                code = self.arm.execute(action_id)
            self.result["code"] = code
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        log(f"  [動作] {label}（ID {action_id}）")
        return "sent"

    def finish(self):
        """話し終わったら呼ぶ。動作の終了を待ち、姿勢を保つ動作なら腕を戻す。"""
        if self.dry or not self.thread:
            return None
        self.thread.join(timeout=12)
        code = self.result.get("code")
        if code != 0:
            log(f"  [動作] 結果: {self.describe(code) if code is not None else '応答なし'}")
        if code == 0 and self.action_id in HOLD_IDS:
            release = self.arm.release()
            if release != 0:
                log(f"  [動作] 腕を戻す(99): {self.describe(release)}")
        self.thread = None
        return code


# ---------------------------------------------------------------- 音声認識 (faster-whisper)
class WhisperSTT:
    def __init__(self, python: Path, model: str, device: str, compute_type: str, dll_dir: str = ""):
        if not Path(python).exists():
            sys.exit(f"faster-whisper の Python が見つかりません: {python}  (--stt-python で指定)")
        log(f"[STT] faster-whisper ({model}, {device}, {compute_type}) を読み込み中… 初回はモデルのダウンロードで時間がかかります")
        extra = ["--dll-dir", dll_dir] if dll_dir else []
        self.proc = subprocess.Popen(
            [str(python), str(HERE / "stt_worker.py"), "--model", model, "--device", device,
             "--compute-type", compute_type, "--language", "ja", "--vad-filter", *extra],
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

    def transcribe(self, wav_path: str) -> str:
        rid = uuid.uuid4().hex
        self.proc.stdin.write(json.dumps({"id": rid, "audioPath": wav_path}) + "\n")
        self.proc.stdin.flush()
        msg = self._read()
        if msg.get("type") == "error":
            raise RuntimeError(f"音声認識エラー: {msg.get('message')}")
        return msg.get("text", "")

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
            if status == "sent":
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
