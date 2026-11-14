#!/usr/bin/env python3
"""R1 操作パネル（ブラウザで操作するチャット＋身ぶり）

  pixi run ui -i Ethernet --volume 100 --gain 9 --motion    # R1 につないで起動 → http://127.0.0.1:8765
  pixi run ui --local --motion                              # R1 なし（PC のスピーカー、身ぶりは表示だけ）

できること:
  - 文字で話しかける / PC のマイクで話しかける（ボタンかスペースキーで録音開始・停止）
  - LLM の返事を R1 が話す。身ぶりオンなら返事に合わせて腕を動かす（Running のときだけ）
  - 「そのまま話させる」: LLM を通さず、入力した文を R1 に話させる
  - 腕の身ぶりボタン（手を振る・拍手・握手など）と「腕を戻す」
  - 人格の切り替え: R1（標準・OpenAI）/ おもしろR1（Z840 の Strata、ローカル LLM）
  - R1 のモード（脱力・直立・Running など）と会話・ログの表示

この PC の中（127.0.0.1）だけで開く。緊急停止はコントローラーの E-Stop で行う（このパネルでは止めない）。
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import tempfile
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "r1_motion"))

import r1_chat as rc   # noqa: E402
import personas         # noqa: E402

# ---------------------------------------------------------------- ログ（画面にも出す）
LOG_LINES: collections.deque[str] = collections.deque(maxlen=400)
_orig_log = rc.log


def log(message: str) -> None:
    _orig_log(message)
    for line in str(message).splitlines() or [""]:
        LOG_LINES.append(f"{time.strftime('%H:%M:%S')} {line}")


rc.log = log   # r1_chat 内（Gesturer など）のログもパネルに出す

# パネルのボタンで送れる腕の動作。会話（LLM）から呼べるのは rc.GESTURES の 6 種類だけ
BUTTONS = [
    ("wave", 25, "手を振る"), ("wave_big", 26, "大きく手を振る"), ("clap", 17, "拍手"),
    ("fist_pump", 29, "ガッツポーズ"), ("hand_on_heart", 33, "右手を胸に"), ("emphasize", 35, "強調"),
    ("shake_hand", 27, "握手の手を出す"), ("high_five", 18, "ハイタッチ"), ("release", 99, "腕を戻す"),
]
BUTTON_IDS = {key: (aid, label) for key, aid, label in BUTTONS}
FSM_NAMES = {0: "脱力（Zero Torque）", 1: "ダンピング", 4: "直立（Locked Stand）", 701: "起き上がり中",
             702: "寝る動作中", 811: "Running（自立）", 816: "Running・腕の動作中"}
GREETING = "こんにちは！僕はR1です。今日は来てくれてありがとう。よろしくお願いします。"


class Engine:
    def __init__(self, args):
        self.args = args
        self.job_lock = threading.Lock()   # 話す・考える仕事は 1 度に 1 つ
        self.busy = ""
        self.history: list[dict] = []
        self.auto_motion = args.motion
        self.fsm = None
        self.fsm_ok = False
        self.rec_stream = None
        self.rec_frames: list[np.ndarray] = []
        self.rec_started = 0.0

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            from credentials import get_secret
            api_key = get_secret()
        if not api_key:
            sys.exit("OpenAI の API キーがありません。`pixi run python r1_chat/credentials.py set` で保存してください")
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

        self.tts = rc.SapiTTS(args.voice, args.rate)
        self.speaker = rc.LocalSpeaker() if args.local else rc.RobotSpeaker(args.interface, args.volume)
        self.gesturer = rc.Gesturer(dry=args.local)
        if not args.local:   # モード読み出しと身ぶりの送信を同じ接続で順番に
            call, lock = self.gesturer.loco._Call, threading.Lock()

            def locked_call(api_id, parameter):
                with lock:
                    return call(api_id, parameter)
            self.gesturer.loco._Call = locked_call
            threading.Thread(target=self._poll_fsm, daemon=True).start()
        # 身ぶりの選択はいつも LLM にさせ、オフのときは送らないだけ（会話の流れを1本に保つ）
        self.brains = {
            "r1": rc.Brain(self.client, args.model, args.reasoning_effort, motion=True),
            "omoshiro": personas.LocalBrain(args.strata_url, personas.PERSONAS["omoshiro"]["prompt"], motion=True,
                                            model=args.strata_model, timeout=120, relay=self._relay(args)),
        }
        self.persona = args.persona
        self.persona_status = {"r1": (True, f"OpenAI {args.model}"), "omoshiro": (False, "確認中")}
        threading.Thread(target=self._poll_personas, daemon=True).start()
        self.stt = None
        if not args.no_stt:
            self.stt = rc.WhisperSTT(Path(args.stt_python), args.stt_model, args.stt_device,
                                     args.stt_compute_type, args.cuda_dll_dir)

    # ------------------------------------------------------------ 状態
    @staticmethod
    def _relay(args):
        if not (args.z840_host and args.z840_user):
            return None
        from ssh_relay import SshRelay
        log(f"[人格] おもしろR1 は SSH 経由で Z840 の Strata を使います（{args.z840_user}@{args.z840_host}）")
        return SshRelay(f"{args.z840_user}@{args.z840_host}", args.z840_key_dir, args.z840_hostkey_alias)

    @property
    def brain(self):
        return self.brains[self.persona]

    def _poll_personas(self) -> None:
        last = None
        while True:
            ok, detail = self.brains["omoshiro"].available()
            self.persona_status["omoshiro"] = (ok, detail)
            if ok != last:
                where = f"SSH {self.args.z840_user}@{self.args.z840_host}" if self.brains["omoshiro"].relay else self.args.strata_url
                log(f"[人格] おもしろR1（Strata / {where}）: {'接続OK ' + detail if ok else '届きません ' + detail}")
                last = ok
            time.sleep(10.0)

    def _poll_fsm(self) -> None:
        last = "unset"
        while True:
            try:
                self.fsm, self.fsm_ok = self.gesturer.loco.get_fsm(), True
            except Exception:
                self.fsm_ok = False
            if self.fsm != last:
                log(f"[モード] {self.fsm} {FSM_NAMES.get(self.fsm, '')}")
                last = self.fsm
            time.sleep(1.0)

    def state(self) -> dict:
        return {
            "local": self.args.local, "fsm": self.fsm, "fsm_ok": self.fsm_ok,
            "fsm_name": FSM_NAMES.get(self.fsm, "不明" if self.fsm is not None else "—"),
            "busy": self.busy, "recording": self.rec_stream is not None,
            "rec_s": round(time.time() - self.rec_started, 1) if self.rec_stream else 0,
            "auto_motion": self.auto_motion, "stt": self.stt is not None,
            "history": self.history[-60:], "log": list(LOG_LINES)[-120:],
            "buttons": [{"key": k, "id": i, "label": l} for k, i, l in BUTTONS],
            "persona": self.persona,
            "personas": [{"key": k, "label": v["label"], "ok": self.persona_status[k][0],
                          "detail": self.persona_status[k][1]} for k, v in personas.PERSONAS.items()],
        }

    def add(self, role: str, text: str, **extra) -> None:
        self.history.append({"role": role, "text": text, "time": time.strftime("%H:%M:%S"), **extra})

    # ------------------------------------------------------------ 仕事の実行
    def submit(self, label: str, fn, *fn_args) -> tuple[bool, str]:
        if not self.job_lock.acquire(blocking=False):
            return False, f"いま「{self.busy}」の途中です"

        def run():
            self.busy = label
            try:
                fn(*fn_args)
            except Exception as error:
                import traceback
                log(f"[エラー] {label}: {type(error).__name__}: {error}\n{traceback.format_exc()}")
                self.add("system", f"エラー: {type(error).__name__}: {error}")
            finally:
                self.busy = ""
                self.job_lock.release()
        threading.Thread(target=run, daemon=True).start()
        return True, "ok"

    def _speak(self, text: str, gesture: str = "none", record: dict | None = None) -> None:
        text = rc.clean_for_speech(text)
        if not text:
            return
        t0 = time.time()
        pcm = rc.boost(self.tts.synth(text), self.args.gain)
        if record is not None:
            record["tts_s"] = round(time.time() - t0, 2)
        status = self.gesturer.start(gesture) if gesture in rc.GESTURES else None
        try:
            self.speaker.play(pcm)
        finally:
            if status == "sent":
                code = self.gesturer.finish()
                if record is not None:
                    record["gesture_code"] = code
        if record is not None and status:
            record["gesture_status"] = status

    def _chat(self, user_text: str, source: str, record: dict | None = None) -> None:
        record = record or {}
        record.update(mode="ui", input=source, user=user_text, persona=self.persona)
        self.add("user", user_text, source=source)
        log(f"あなた（{source}） > {user_text}")
        t0 = time.time()
        try:
            answer, gesture = self.brain.reply(user_text)
        except Exception as error:
            log(f"  [LLM エラー] {type(error).__name__}: {str(error)[:200]}")
            record["error"] = f"llm: {type(error).__name__}: {str(error)[:200]}"
            answer, gesture = "ごめんなさい、今ちょっと通信がうまくいきません。もう一度話しかけてください。", "none"
        record.update(answer=answer, llm_s=round(time.time() - t0, 2), gesture=gesture)
        send = gesture if self.auto_motion else "none"
        label = rc.GESTURES[gesture][1] if gesture in rc.GESTURES else ""
        if label and not self.auto_motion:
            label += "（身ぶりオフのため送らない）"
        self.add("r1", answer, gesture=label, llm_s=record["llm_s"], persona=personas.PERSONAS[self.persona]["label"])
        log(f"R1 ({record['llm_s']:.1f}s) > {answer}" + (f"  ［身ぶり: {label}］" if label else ""))
        try:
            self._speak(answer, send, record)
        except Exception as error:
            log(f"  [発話エラー] {type(error).__name__}: {error}")
            record["error"] = f"speak: {type(error).__name__}: {error}"
        rc.log_turn(**record)

    def _say(self, text: str) -> None:
        self.add("r1", text, gesture="", direct=True)
        log(f"R1（そのまま） > {text}")
        self._speak(text)

    def _greet(self) -> None:
        self.add("r1", GREETING, gesture=rc.GESTURES["wave"][1], direct=True)
        log(f"R1（あいさつ） > {GREETING}")
        self._speak(GREETING, "wave")

    # ------------------------------------------------------------ 入口
    def chat(self, text: str):
        return self.submit("考え中", self._chat, text, "文字")

    def say(self, text: str):
        return self.submit("発話中", self._say, text)

    def greet(self):
        return self.submit("あいさつ中", self._greet)

    def gesture(self, key: str) -> tuple[bool, str]:
        if key not in BUTTON_IDS:
            return False, "不明な動作です"
        aid, label = BUTTON_IDS[key]
        if self.args.local:
            log(f"[動作] {label}（ID {aid}、--local なので送らない）")
            return True, "ok"
        if aid != 99 and self.fsm not in rc.STANDING_FSM:
            return False, f"立っていない（モード {self.fsm}）ので送りません"
        if self.gesturer.thread and self.gesturer.thread.is_alive():
            return False, "前の身ぶりがまだ動いています"

        def run():
            code = self.gesturer.arm.execute(aid)
            if code == 7401 and aid != 99:   # 握手などで腕が止まったまま → 戻してから送り直す
                log(f"[動作] 腕が姿勢保持中なので、戻してから {label} を送り直します")
                self.gesturer.arm.release()
                code = self.gesturer.arm.execute(aid)
            log(f"[動作] {label}（ID {aid}） → {self.gesturer.describe(code)}")
            if code != 0:
                self.add("system", f"{label}: {self.gesturer.describe(code)}")
        # 会話の自動身ぶりと同じ「動作中」として扱い、重ねて送らない
        thread = threading.Thread(target=run, daemon=True)
        self.gesturer.thread, self.gesturer.action_id = thread, None
        thread.start()
        log(f"[動作] {label}（ID {aid}）を送信")
        return True, "ok"

    def set_persona(self, key: str):
        if key not in self.brains:
            return False, "不明な人格です"
        if self.job_lock.locked():
            return False, f"いま「{self.busy}」の途中です"
        ok, detail = self.persona_status[key]
        if key == "omoshiro" and not ok:
            ok, detail = self.brains[key].available()
            self.persona_status[key] = (ok, detail)
            if not ok:
                return False, f"Z840 の Strata に届きません（{detail}）"
        self.persona = key
        label = personas.PERSONAS[key]["label"]
        self.add("system", f"人格を「{label}」に切り替えました")
        log(f"[人格] {label} に切り替え")
        return True, "ok"

    def reset(self):
        brain = self.brain
        if hasattr(brain, "reset"):
            brain.reset()
        else:
            brain.previous_id = None
        self.add("system", "会話をリセットしました")
        log("[会話] リセット")
        return True, "ok"

    def set_motion(self, on: bool):
        self.auto_motion = bool(on)
        log(f"[身ぶり] 会話の身ぶりを{'オン' if on else 'オフ'}にしました")
        return True, "ok"

    def record_start(self):
        import sounddevice as sd
        if not self.stt:
            return False, "音声認識なし（--no-stt）で起動しています"
        if self.rec_stream is not None:
            return False, "録音中です"
        if self.job_lock.locked():
            return False, f"いま「{self.busy}」の途中です"
        mic = self.args.mic_device
        mic = int(mic) if mic and str(mic).isdigit() else mic
        self.rec_frames = []
        stream = sd.InputStream(samplerate=rc.SAMPLE_RATE, channels=1, dtype="int16", device=mic,
                                callback=lambda data, *_: self.rec_frames.append(data.copy()))
        stream.start()
        self.rec_stream, self.rec_started = stream, time.time()
        log("[録音] 開始")
        return True, "ok"

    def record_stop(self):
        stream, self.rec_stream = self.rec_stream, None
        if stream is None:
            return False, "録音していません"
        stream.stop()
        stream.close()
        audio = np.concatenate(self.rec_frames)[:, 0] if self.rec_frames else np.zeros(0, dtype=np.int16)
        log(f"[録音] 停止 ({len(audio) / rc.SAMPLE_RATE:.1f}s)")
        if len(audio) < rc.SAMPLE_RATE * 0.3:
            return False, "短すぎるので無視しました"
        return self.submit("聞き取り中", self._listen, audio)

    def _listen(self, audio: np.ndarray) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            with wave.open(path, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(rc.SAMPLE_RATE)
                w.writeframes(audio.astype("<i2").tobytes())
            t0 = time.time()
            text = self.stt.transcribe(path)
        finally:
            os.unlink(path)
        stt_s = round(time.time() - t0, 2)
        log(f"  認識 ({stt_s:.1f}s): {text or '（聞き取れませんでした）'}")
        if not text:
            self.add("system", "聞き取れませんでした。もう一度どうぞ")
            return
        self.busy = "考え中"
        self._chat(text, "声", {"stt_s": stt_s, "audio_s": round(len(audio) / rc.SAMPLE_RATE, 2)})


# ---------------------------------------------------------------- HTTP
PAGE = (HERE / "web_ui.html").read_text(encoding="utf-8")


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):   # アクセスログは出さない
            pass

        def _json(self, data, status=200):
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/state":
                self._json(engine.state())
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                data = {}
            text = str(data.get("text", "")).strip()[:300]
            routes = {
                "/api/chat": lambda: engine.chat(text) if text else (False, "文が空です"),
                "/api/say": lambda: engine.say(text) if text else (False, "文が空です"),
                "/api/greet": engine.greet,
                "/api/gesture": lambda: engine.gesture(str(data.get("key", ""))),
                "/api/record/start": engine.record_start,
                "/api/record/stop": engine.record_stop,
                "/api/reset": engine.reset,
                "/api/motion": lambda: engine.set_motion(data.get("on")),
                "/api/persona": lambda: engine.set_persona(str(data.get("key", ""))),
            }
            if self.path not in routes:
                return self._json({"error": "not found"}, 404)
            try:
                ok, message = routes[self.path]()
            except Exception as error:
                ok, message = False, f"{type(error).__name__}: {error}"
                log(f"[エラー] {self.path}: {message}")
            self._json({"ok": ok, "message": message}, 200 if ok else 409)
    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description="R1 操作パネル（ブラウザ）")
    p.add_argument("-i", "--interface", help="R1 につないだ有線LANアダプタ名（例: Ethernet）")
    p.add_argument("--local", action="store_true", help="R1 を使わず PC のスピーカーで鳴らす")
    p.add_argument("--motion", action="store_true", help="会話の身ぶりをオンで始める（パネルで切り替え可）")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--volume", type=int, default=100)
    p.add_argument("--gain", type=float, default=9.0)
    p.add_argument("--voice", default="Microsoft Ichiro")
    p.add_argument("--rate", type=int, default=0)
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5-mini"))
    p.add_argument("--reasoning-effort", default="minimal")
    p.add_argument("--persona", choices=list(personas.PERSONAS), default="r1", help="起動時の人格")
    p.add_argument("--strata-url", default=os.environ.get("R1_STRATA_URL", "http://127.0.0.1:8081/v1"),
                   help="Strata（OpenAI 互換）の /v1 の URL。Z840 へは scripts/strata_tunnel.ps1 のトンネル経由")
    p.add_argument("--strata-model", default="", help="Strata のモデル名（空なら /v1/models の先頭）")
    p.add_argument("--z840-host", default=os.environ.get("R1_Z840_HOST", ""), help="Z840 の SSH 先（Tailscale の IP など）")
    p.add_argument("--z840-user", default=os.environ.get("R1_Z840_USER", ""))
    p.add_argument("--z840-key-dir", default=os.environ.get("R1_Z840_KEY_DIR", ""), help="id_ed25519 と known_hosts のフォルダ")
    p.add_argument("--z840-hostkey-alias", default=os.environ.get("R1_Z840_HOSTKEY_ALIAS", ""))
    p.add_argument("--no-stt", action="store_true", help="音声認識を読み込まない（文字だけ）")
    p.add_argument("--stt-python", default=os.environ.get("R1_STT_PYTHON", str(rc.DEFAULT_STT_PYTHON)))
    p.add_argument("--stt-model", default="small")
    p.add_argument("--stt-device", default="cuda")
    p.add_argument("--stt-compute-type", default="float16")
    p.add_argument("--cuda-dll-dir", default=rc.DEFAULT_CUDA_DLL_DIR)
    p.add_argument("--mic-device", default=None)
    p.add_argument("--log-dir", default=str(rc.REPO_ROOT / "logs"))
    args = p.parse_args()
    if not args.local and not args.interface:
        p.error("-i <有線LANアダプタ名> か --local のどちらかを指定してください")

    base = rc.setup_logging(args.log_dir, "ui")
    log(f"[log] {base}.log / {base.name}_turns.jsonl")
    engine = Engine(args)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine))
    log(f"\n=== R1 操作パネル: http://127.0.0.1:{args.port} を開いてください（Ctrl+C で終了） ===")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("終了します")
    finally:
        server.server_close()
        if engine.stt:
            engine.stt.close()


if __name__ == "__main__":
    main()
