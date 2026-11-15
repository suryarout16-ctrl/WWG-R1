#!/usr/bin/env python3
"""R1 操作パネル（ブラウザで操作するチャット＋身ぶり）

  pixi run ui -i Ethernet --volume 100 --gain 9 --motion    # R1 につないで起動 → http://127.0.0.1:8765
  pixi run ui --local --motion                              # R1 なし（PC のスピーカー、身ぶりは表示だけ）

できること:
  - 文字で話しかける / PC のマイクで話しかける（ボタンかスペースキーで録音開始・停止）
  - LLM の返事を R1 が話す。身ぶりオンなら返事に合わせて腕を動かす（Running のときだけ）
  - 「そのまま話させる」: LLM を通さず、入力した文を R1 に話させる
  - 腕の身ぶりボタン（手を振る・拍手・握手など）と「腕を戻す」
  - 新人研修を担当する熟練工R1（Strata）に固定、通話形式で自動の聞き取り・返答
  - 台本・作業指示書の取り込み（テキスト・Markdown・本文の貼り付け）
  - R1 のモード（脱力・直立・Running など）と会話・ログの表示

この PC の中（127.0.0.1）だけで開く。緊急停止はコントローラーの E-Stop で行う（このパネルでは止めない）。
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import os
import secrets
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
from training_materials import TrainingMaterials  # noqa: E402
from voice_call import VoiceCall  # noqa: E402
from training_flow import TrainingFlow, is_training_start  # noqa: E402
from body_control import BodyController, requested_body_actions  # noqa: E402
from posture_control import StandController  # noqa: E402
from posture_intent import parse_stand_request  # noqa: E402
import bilingual  # noqa: E402
import inspect  # noqa: E402
import ipaddress  # noqa: E402
from dds_network import configure_dds_address  # noqa: E402
from audio_input import resolve_input_device, refresh_audio_devices  # noqa: E402

BODY_DIRECTIONS = {"move_forward": "forward", "move_backward": "backward",
                   "move_left": "left", "move_right": "right",
                   "turn_left": "turn_left", "turn_right": "turn_right", "stop_move": "stop"}

# ---------------------------------------------------------------- ログ（画面にも出す）
LOG_LINES: collections.deque[str] = collections.deque(maxlen=400)
_orig_log = rc.log


def log(message: str) -> None:
    _orig_log(message)
    for line in str(message).splitlines() or [""]:
        LOG_LINES.append(f"{time.strftime('%H:%M:%S')} {line}")


rc.log = log   # r1_chat 内（Gesturer など）のログもパネルに出す

# パネルからの手動操作。会話（LLM）は rc.GESTURES の許可された身ぶりだけ。
BUTTONS = [
    ("wave", 25, "手を振る"), ("wave_big", 26, "大きく手を振る"), ("clap", 17, "拍手"),
    ("fist_pump", 29, "ガッツポーズ"), ("hand_on_heart", 33, "右手を胸に"), ("emphasize", 35, "強調"),
    ("shake_hand", 27, "握手の手を出す"), ("high_five", 18, "ハイタッチ"), ("release", 99, "腕を戻す"),
    ("hat_tip", None, "帽子の身ぶり（教示動作）"),
]
BUTTON_IDS = {key: (aid, label) for key, aid, label in BUTTONS}
FSM_NAMES = {0: "脱力（Zero Torque）", 1: "ダンピング", 4: "直立（Locked Stand）", 701: "起き上がり中",
             702: "寝る動作中", 811: "Running（自立）", 816: "Running・腕の動作中"}
GREETING = personas.SELF_INTRODUCTION


def _forward_settings(args) -> tuple[float, float]:
    """前進だけの明示設定を、デバイス初期化より先に検査する。"""
    speed, duration = getattr(args, "forward_speed", 0.1), getattr(args, "forward_duration", 1.0)
    for name, value, maximum in (("forward_speed", speed, 0.3), ("forward_duration", duration, 2.0)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 < value <= maximum):
            raise ValueError(f"--{name.replace('_', '-')}は0より大きく{maximum:g}以下にしてください")
    return float(speed), float(duration)


def _strata_timeout(args) -> float:
    value = getattr(args, "strata_timeout", 10.0)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 1 <= value <= 120):
        raise ValueError("--strata-timeoutは1秒以上120秒以下にしてください")
    return float(value)


def _hold_settings(args) -> tuple[float, float]:
    linear = getattr(args, "hold_linear_speed", 0.9)
    yaw = getattr(args, "hold_yaw_speed", 1.8)
    for name, value, maximum in (("hold_linear_speed", linear, 0.9), ("hold_yaw_speed", yaw, 1.8)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 < value <= maximum):
            raise ValueError(f"--{name.replace('_', '-')}は0より大きく{maximum:g}以下にしてください")
    return float(linear), float(yaw)


class Engine:
    def __init__(self, args):
        self.forward_speed, self.forward_duration = _forward_settings(args)
        self.hold_linear_speed, self.hold_yaw_speed = _hold_settings(args)
        self.strata_timeout = _strata_timeout(args)
        selected_backend = getattr(args, "mentor_backend", "strata")
        if selected_backend not in ("strata", "openai"):
            raise ValueError("--mentor-backendはstrata/openaiです")
        self.args = args
        if args.local and getattr(args, "dds_address", None):
            raise ValueError("ローカルプレビューでは--dds-addressを指定しないでください")
        self.dds_address = configure_dds_address(getattr(args, "dds_address", None))
        self.job_lock = threading.Lock()   # 話す・考える仕事は 1 度に 1 つ
        self.control_lock = threading.RLock()
        self.busy = ""
        self.history: list[dict] = []
        self.auto_motion = args.motion
        self.fsm = None
        self.fsm_ok = False
        self.rec_stream = None
        self.rec_frames: list[np.ndarray] = []
        self.rec_started = 0.0
        self.call = None
        self.closed = False
        self.posture_epoch = 0
        self.hold_epoch = secrets.randbits(48) + 1
        self._hold_token = None
        self.training = TrainingFlow()
        self.materials = TrainingMaterials(Path(args.materials_file))
        self.client = None
        self.mentor_openai_client = None
        self.mentor_backend = "strata"
        self.mentor_backends = {}
        # GPT指定なのに資格情報が無い場合、音声やロボットを初期化する前に止める。
        if selected_backend == "openai":
            self._gpt_mentor()

        self.tts = rc.SapiTTS(args.voice, args.rate)
        self.tts_en = None   # 英語の返事を読む声。最初の英語の返事のときに用意する
        self.speaker = rc.LocalSpeaker() if args.local else rc.RobotSpeaker(args.interface, args.volume)
        self.gesturer = rc.Gesturer(dry=args.local, hat_action=args.hat_action, hat_fallback=args.hat_fallback)
        self.body_enabled = bool(args.local or getattr(args, "body_control", False))
        self.body_feedback = None
        if self.body_enabled and not args.local:
            from body_feedback import MotorFeedbackMonitor
            # Python版CRCの負荷を抑え、RPC応答を扱うスレッドにも余裕を残す。
            self.body_feedback = MotorFeedbackMonitor(sample_hz=10).start()
        self.body = BodyController(None if args.local else self.gesturer.loco, dry=args.local,
                                   forward_speed=self.forward_speed,
                                   motion_lock=self.gesturer.lock,
                                   velocity_127_compatible=getattr(args, "velocity_127_compatible", False),
                                   feedback=self.body_feedback.snapshot if self.body_feedback else None,
                                   require_feedback=self.body_enabled and not args.local,
                                   on_event=lambda event: log("[歩行] " + json.dumps(event, ensure_ascii=False)))
        if not args.local and getattr(args, "require_stop_verification", False):
            self.body.require_stop_verification("起動後の停止確認が必要です。新しい状態データで停止を再確認してください")
        self.posture = StandController(None if args.local else self.gesturer.loco, dry=args.local,
                                       motion_lock=self.gesturer.lock,
                                       on_event=lambda event: log("[起立] " + json.dumps(event, ensure_ascii=False)))
        if not args.local:   # モード読み出しと身ぶりの送信を同じ接続で順番に
            call, lock = self.gesturer.loco._Call, threading.Lock()

            def locked_call(api_id, parameter):
                with lock:
                    return call(api_id, parameter)
            self.gesturer.loco._Call = locked_call
            threading.Thread(target=self._poll_fsm, daemon=True).start()
        # 身ぶりの選択はいつも LLM にさせ、オフのときは送らないだけ（会話の流れを1本に保つ）
        self.brains = {
            "mentor": personas.LocalBrain(args.strata_url, personas.PERSONAS["mentor"]["prompt"], motion=True,
                                            model=args.strata_model, timeout=self.strata_timeout, relay=self._relay(args)),
        }
        self.mentor_backends["strata"] = self.brains["mentor"]
        self.brains["mentor"].set_training_context(self.materials.context())
        if hasattr(self.brains["mentor"], "set_body_enabled"):
            self.brains["mentor"].set_body_enabled(self.body_enabled)
        self.persona = personas.normalize_persona_key(args.persona)
        self.persona_status = {"r1": (False, "明示的に切り替えたときだけ接続"), "mentor": (False, "確認中")}
        if selected_backend != "strata":
            ok, message = self.set_mentor_backend(selected_backend)
            if not ok:
                raise RuntimeError(message)
        if self.persona == "r1":
            self._openai_brain()
        threading.Thread(target=self._poll_personas, daemon=True).start()
        self.stt = None
        if not args.no_stt:
            self.stt = rc.WhisperSTT(Path(args.stt_python), args.stt_model, args.stt_device,
                                     args.stt_compute_type, args.cuda_dll_dir,
                                     language=getattr(args, "stt_language", "ja"))
        mic = args.mic_device
        mic = int(mic) if mic and str(mic).isdigit() else mic
        self.call = VoiceCall(self._call_utterance, can_listen=self._can_listen,
                              on_error=self._call_error, device=mic,
                              auto_reconnect=mic is not None,
                              device_refresher=refresh_audio_devices if not args.local and mic is not None else None,
                              threshold=args.call_threshold, silence_seconds=args.call_silence)

    def _openai_brain(self):
        if "r1" in self.brains:
            return self.brains["r1"]
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            from credentials import get_secret
            api_key = get_secret()
        if not api_key:
            raise RuntimeError("標準R1を使うにはOpenAIのAPIキーが必要です")
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.brains["r1"] = rc.Brain(self.client, self.args.model, self.args.reasoning_effort, motion=True)
        self.persona_status["r1"] = (True, f"OpenAI {self.args.model}")
        return self.brains["r1"]

    def _gpt_mentor(self):
        if "openai" in self.mentor_backends:
            return self.mentor_backends["openai"]
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            from credentials import get_secret
            api_key = get_secret()
        if not api_key:
            raise RuntimeError("GPT APIへ切り替えるにはOpenAIのAPIキーが必要です")
        from openai import OpenAI
        from mentor_backend import OpenAIMentorBrain
        # 自動再試行によって本番の待ち時間が伸びるのを避ける。
        try:
            client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1",
                            timeout=20.0, max_retries=0)
        except Exception:
            raise RuntimeError("GPT APIクライアントを初期化できません") from None
        brain = OpenAIMentorBrain(client, self.args.model, self.args.reasoning_effort)
        self.mentor_openai_client = client
        self.mentor_backends["openai"] = brain
        return brain

    def _persona_label(self, key):
        if key == "mentor":
            return "研修担当R1（GPT API）" if self.mentor_backend == "openai" else "研修担当R1（Strata）"
        return personas.PERSONAS[key]["label"]

    def set_mentor_backend(self, key):
        """人格は変えず、待機中に推論先だけ切り替える。会話と資料は引き継ぐ。"""
        if key not in ("strata", "openai"):
            return False, "不明な応答先です"
        with self.control_lock:
            if self.job_lock.locked() or self.rec_stream is not None or self._call_active():
                return False, "通話・録音・処理が終わってから応答先を切り替えてください"
            if key == self.mentor_backend:
                return True, "ok"
            source = self.brains["mentor"]
            try:
                target = self._gpt_mentor() if key == "openai" else self.mentor_backends["strata"]
            except Exception:
                return False, "GPT APIを準備できません。APIキーとモデルの設定を確認してください"
            target.set_training_context(source.training_context)
            target.set_runtime_context(source.runtime_context)
            target.set_body_enabled(source.body_enabled)
            target.history = copy.deepcopy(source.history[-target.MAX_TURNS * 2:])
            target.last_reply = None
            self.brains["mentor"] = target
            self.mentor_backend = key
            self.persona_status["mentor"] = (False, "接続確認中")
            label = self._persona_label("mentor")
            self.add("system", f"応答先を{label}へ切り替えました。人格・教材・会話は引き継いでいます")
            log(f"[応答先] {label} model={target.model or '未取得'}")
            return True, "ok"

    # ------------------------------------------------------------ 状態
    @staticmethod
    def _relay(args):
        if not (args.z840_host and args.z840_user):
            return None
        from ssh_relay import SshRelay
        log("[人格] 研修担当R1はSSH経由でStrataを使います")
        return SshRelay(f"{args.z840_user}@{args.z840_host}", args.z840_key_dir, args.z840_hostkey_alias)

    @property
    def brain(self):
        return self.brains[self.persona]

    def _poll_personas(self) -> None:
        last = None
        while not self.closed:
            brain, backend = self.brains["mentor"], self.mentor_backend
            ok, detail = brain.available()
            # 切り替え前の遅いStrata確認結果で、新しいGPT側の表示を上書きしない。
            with self.control_lock:
                if brain is self.brains["mentor"]:
                    self.persona_status["mentor"] = (ok, detail)
                    if (backend, ok) != last:
                        log(f"[人格] {self._persona_label('mentor')}: {'接続OK ' + detail if ok else '届きません ' + detail}")
                        last = (backend, ok)
            time.sleep(10.0)

    def _poll_fsm(self) -> None:
        last = "unset"
        last_ok = None
        while not self.closed:
            try:
                self.fsm, self.fsm_ok = self.gesturer.loco.get_fsm(), True
                if last_ok is False:
                    log("[モード] 状態の読み取りが復旧しました")
            except Exception as error:
                if last_ok is not False:
                    log(f"[モード] 状態を読めないため新しい歩行を見送ります: {type(error).__name__}: {error}")
                self.fsm_ok = False
            last_ok = self.fsm_ok
            if self.fsm != last:
                log(f"[モード] {self.fsm} {FSM_NAMES.get(self.fsm, '')}")
                last = self.fsm
            time.sleep(1.0)

    @property
    def connection_kind(self) -> str:
        if self.args.local:
            return "local"
        if self.dds_address:
            return "virtual" if ipaddress.IPv4Address(self.dds_address).is_loopback else "robot"
        if getattr(self.args, "interface", None) == "Loopback Pseudo-Interface 1":
            return "virtual"
        return "robot"

    def state(self) -> dict:
        return {
            "local": self.args.local, "fsm": self.fsm, "fsm_ok": self.fsm_ok,
            "virtual": self.connection_kind == "virtual", "connection_kind": self.connection_kind,
            "dds_address": self.dds_address,
            "voice": getattr(self.tts, "voice_name", self.args.voice),
            "voice_en": getattr(self.tts_en, "voice_name", None),
            "voice_gain_db": self.args.gain,
            "fsm_name": FSM_NAMES.get(self.fsm, "不明" if self.fsm is not None else "—"),
            "busy": self.busy, "recording": self.rec_stream is not None,
            "rec_s": round(time.time() - self.rec_started, 1) if self.rec_stream else 0,
            "auto_motion": self.auto_motion, "stt": self.stt is not None,
            "call": self.call.state() if self.call else {"active": False},
            "training": self.training.state(),
            "posture": {**self.posture.state(), "enabled": self.body_enabled},
            "persona_locked": not self.args.allow_persona_switch,
            "materials": self.materials.documents(),
            "hat_action": self.args.hat_action, "hat_fallback": self.args.hat_fallback,
            "body": {**self.body.state(), "enabled": self.body_enabled,
                     "forward_duration": self.forward_duration, "hold_epoch": self.hold_epoch,
                     "monitor": self._feedback_monitor_state(),
                     "hold_profile": {"linear_speed": self.hold_linear_speed, "yaw_speed": self.hold_yaw_speed,
                                      "heartbeat_ms": 200, "lease_ms": 750}},
            "history": self.history[-60:], "log": list(LOG_LINES)[-120:],
            "buttons": [{"key": k, "id": i, "label": l} for k, i, l in BUTTONS],
            "persona": self.persona,
            "mentor_backend": self.mentor_backend,
            "mentor_model": self.brains["mentor"].model,
            "strata_timeout": self.strata_timeout,
            "personas": [{"key": k, "label": self._persona_label(k), "ok": self.persona_status[k][0],
                          "detail": self.persona_status[k][1]} for k, v in personas.PERSONAS.items()],
        }

    def _feedback_monitor_state(self):
        reporter = getattr(self.body_feedback, "diagnostics", None)
        if not callable(reporter):
            return None
        try:
            result = reporter()
            return result if isinstance(result, dict) else None
        except Exception as error:
            return {"recovering": True, "last_error": type(error).__name__}

    def add(self, role: str, text: str, **extra) -> None:
        self.history.append({"role": role, "text": text, "time": time.strftime("%H:%M:%S"), **extra})

    # ------------------------------------------------------------ 仕事の実行
    def submit(self, label: str, fn, *fn_args) -> tuple[bool, str]:
        with self.control_lock:
            if self.rec_stream is not None:
                return False, "録音を停止してから操作してください"
            if not self.job_lock.acquire(blocking=False):
                return False, f"いま「{self.busy}」の途中です"
            self.busy = label

        def run():
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

    def _speak(self, text: str, gesture: str = "none", record: dict | None = None,
               should_continue=None, allowed_body=None) -> None:
        current = lambda: not self.closed and (should_continue is None or should_continue())
        text = rc.clean_for_speech(text)
        if not text or not current():
            return
        t0 = time.time()
        pcm = rc.boost(self._tts_for(text).synth(text), self.args.gain)
        if not current():
            return
        if record is not None:
            record["tts_s"] = round(time.time() - t0, 2)
        status, body_started = None, False
        if gesture in BODY_DIRECTIONS:
            if gesture not in (allowed_body or set()):
                status = "body_not_requested"
                log("[歩行] 利用者の直接の移動依頼がないため、LLMの移動選択を送りません")
            elif not self.body_enabled:
                status = "body_disabled"
            elif self.gesturer.thread is not None:
                status = "body_arms_busy"
            else:
                with self.control_lock:
                    if not current():
                        return
                    ok, message = self.body.move(BODY_DIRECTIONS[gesture], self._body_duration(gesture))
                body_started = ok and gesture != "stop_move"
                status = "body_preview" if ok and self.args.local else "body_sent" if ok else "body_rejected"
                log(f"[歩行] {message}")
        elif gesture in rc.GESTURES:
            status = self.gesturer.start(gesture)
        try:
            self.speaker.play(pcm, should_continue=current)
        finally:
            if body_started:
                if self.body.state()["busy"] or self.body.state()["stop_required"]:
                    self.body.stop()
                self.body.wait_until_idle()
                if record is not None:
                    record["body"] = self.body.state()
            if status and status.startswith("sent"):
                code = self.gesturer.finish()
                if record is not None:
                    record["gesture_code"] = code
        if record is not None and status:
            record["gesture_status"] = status

    def _tts_for(self, text: str):
        """英語の文は英語の声（既定 Microsoft David）、それ以外は日本語の声で読む。"""
        if rc.text_language(text) != "en":
            return self.tts
        if self.tts_en is None:
            voice = getattr(self.args, "voice_en", "Microsoft David")
            try:
                self.tts_en = rc.SapiTTS(voice, self.args.rate)
            except Exception as error:
                log(f"[声] 英語の声 {voice} を使えないので日本語の声で読みます: {error}")
                self.tts_en = self.tts
        return self.tts_en

    def _reply(self, user_text: str, language: str, low_confidence: bool,
               allowed_gestures=None, max_chars=None):
        """LocalBrain（研修担当R1）には言語と確信度を渡す。古い Brain・テスト用の偽物は従来どおり。"""
        try:
            params = inspect.signature(self.brain.reply).parameters
        except (TypeError, ValueError):
            params = {}
        kwargs = {}
        if "language" in params:
            kwargs.update(language=language, low_confidence=low_confidence)
        if "allowed_gestures" in params and allowed_gestures is not None:
            kwargs["allowed_gestures"] = allowed_gestures
        if "max_chars" in params and max_chars is not None:
            kwargs["max_chars"] = max_chars
        return self.brain.reply(user_text, **kwargs)

    def _chat(self, user_text: str, source: str, record: dict | None = None,
              should_continue=None, language: str | None = None, low_confidence: bool = False,
              training_step: str | None = None, display_user: bool = True) -> dict | None:
        if (training_step is None and self.persona == "mentor" and not low_confidence
                and is_training_start(user_text)):
            return self._run_training(user_text, source, should_continue, language)
        stand_request = parse_stand_request(user_text) if training_step is None else None
        if stand_request is not None:
            self.add("user", user_text, source=source)
            if low_confidence or stand_request["posture"] != "supine" or not stand_request["ready"]:
                message = ("Select the supine posture and confirm preparation in the stand-up panel first."
                           if language == "en" or rc.text_language(user_text) == "en" else
                           "起立欄で仰向けと準備確認を選んでください。チャットなら「仰向けで準備確認済み、起立して」と明示してください。")
                self.add("r1", message)
                self._speak(message, should_continue=should_continue)
                return {"posture_rejected": True}
            return self._run_stand(stand_request["posture"], stand_request["ready"], should_continue,
                                   self.posture_epoch)
        generation = self.training.generation if self.training.active else None
        current = lambda: (not self.closed and (should_continue is None or should_continue())
                           and (generation is None or generation == self.training.generation))
        if not current():
            return
        record = record if record is not None else {}
        language = language if language in ("ja", "en") else rc.text_language(user_text)
        record.update(mode="ui", input=source, user=user_text, persona=self.persona, language=language)
        if low_confidence:
            record["low_confidence"] = True
        if self.persona == "mentor":
            record.update(backend=self.mentor_backend, model=self.brain.model)
        if self.training.active:
            record.update(training_phase=training_step or self.training.phase,
                          training_generation=self.training.generation)
        if display_user:
            self.add("user", user_text, source=source)
        log(f"あなた（{source}） > {user_text}")
        t0 = time.time()
        try:
            if hasattr(self.brain, "set_runtime_context"):
                if self.connection_kind == "local":
                    mode = "R1未接続のローカルプレビュー。動作は実機へ送らない。"
                elif self.connection_kind == "virtual":
                    mode = f"PC内の仮想R1へ接続。物理動作・距離・実機の成功を検証するものではない。読取FSM={self.fsm}、状態読取成功={self.fsm_ok}。"
                else:
                    mode = f"実機R1接続モード。読取FSM={self.fsm}、状態読取成功={self.fsm_ok}。"
                if self.persona == "mentor":
                    mode += "現在の応答生成は" + ("GPT API。" if self.mentor_backend == "openai" else "Z840のStrata。")
                mode += (("短時間の歩行・旋回を許可。" if self.body_enabled else "歩行の実行は無効。") +
                         "新規の低レベル関節指令は未検証のため保留。工場の機械を直接操作する機能はない。")
                if self.training.active:
                    mode += self.training.context(training_step)
                    if training_step:
                        choices = self._training_actions(training_step)
                        mode += "\nこの段階で選べるgestureは" + ", ".join(sorted(choices)) + "のみ。"
                        if "move_forward" in choices:
                            mode += ("研修開始操作にひもづく仮想の見本歩行許可がある。"
                                     "move_forwardで荷物の移動を歩行で表現し、発話は移動前の短い説明にする。")
                        else:
                            mode += "この段階では歩行を追加しない。許可された腕の身ぶりで説明する。"
                self.brain.set_runtime_context(mode)
            model_input = self.training.request(training_step) if training_step else user_text
            if training_step:
                record["model_input"] = model_input
            answer, gesture = self._reply(model_input, language, low_confidence,
                                          allowed_gestures=self._training_actions(training_step) if training_step else None,
                                          max_chars=(180 if language == "en" else 100) if training_step else None)
        except Exception as error:
            log(f"  [LLM エラー] {type(error).__name__}: {str(error)[:200]}")
            record["error"] = f"llm: {type(error).__name__}: {str(error)[:200]}"
            answer, gesture = bilingual.ERROR_REPLY[language], "none"
        if not current():
            record.update(cancelled=True, stage="llm")
            rc.log_turn(**record)
            return
        if training_step and gesture not in self._training_actions(training_step):
            record["training_action_blocked"] = gesture
            gesture = "none"
        record.update(answer=answer, llm_s=round(time.time() - t0, 2), gesture=gesture)
        if self.persona == "mentor":
            record["model"] = self.brain.model
        model_reply = getattr(self.brain, "last_reply", None)
        if isinstance(model_reply, dict):
            record["model_reply"] = dict(model_reply)
        send = gesture if self.auto_motion else "none"
        body_actions = getattr(personas, "BODY_ACTIONS", {})
        label = rc.GESTURES[gesture][1] if gesture in rc.GESTURES else body_actions.get(gesture, "")
        if label and not self.auto_motion:
            label += "（身ぶりオフのため送らない）"
        self.add("r1", answer, gesture=label, llm_s=record["llm_s"], persona=self._persona_label(self.persona))
        log(f"R1 ({record['llm_s']:.1f}s) > {answer}" + (f"  ［身ぶり: {label}］" if label else ""))
        try:
            allowed_body = requested_body_actions(user_text) | bilingual.requested_body_actions_en(user_text)
            if training_step:
                allowed_body = self._training_actions(training_step) & set(BODY_DIRECTIONS)
            self._speak(answer, send, record, should_continue=current, allowed_body=allowed_body)
        except Exception as error:
            log(f"  [発話エラー] {type(error).__name__}: {error}")
            record["error"] = f"speak: {type(error).__name__}: {error}"
        rc.log_turn(**record)
        if self.training.active and training_step is None and current() and not record.get("error"):
            self.training.learner_replied()
        return record

    def _training_actions(self, phase: str) -> set[str]:
        if phase == "demonstrate" and self.connection_kind in ("local", "virtual") and self.body_enabled:
            return {"none", "move_forward"}
        if phase in ("explain", "demonstrate"):
            return {"none", "point_forward", "emphasize"}
        return {"none", *(name for name in rc.GESTURES if name != "hat_tip")}

    def _run_training(self, user_text: str, source: str, should_continue=None, language=None):
        if self.persona != "mentor":
            return None
        if not self.auto_motion:
            self.add("system", "見本を動かすため、会話の身ぶりをオンにしてから研修を開始してください")
            return None
        with self.control_lock:
            generation = self.training.start()
        current = lambda: (not self.closed and generation == self.training.generation
                           and (should_continue is None or should_continue()))
        result = None
        try:
            for step in self.training.STEPS:
                with self.control_lock:
                    if not current():
                        return result
                    self.training.enter(step)
                result = self._chat(user_text, source if step == "explain" else "研修進行",
                                    record={"training_phase": step}, should_continue=current,
                                    language=language, training_step=step, display_user=step == "explain")
                if not current():
                    return result
                status = (result or {}).get("gesture_status", "")
                if (not result or result.get("error") or result.get("training_action_blocked")
                        or (step == "demonstrate" and result.get("gesture") == "none")
                        or (status and status not in ("sent", "dry", "body_sent", "body_preview"))
                        or result.get("gesture_code", 0) != 0
                        or (result.get("body") or {}).get("error")):
                    self.training.stop("error")
                    self.add("system", "研修を中断しました。応答・動作の結果を確認してください")
                    return result
            with self.control_lock:
                if current():
                    self.training.waiting()
            return result
        finally:
            if generation == self.training.generation and self.training.phase in self.training.STEPS:
                self.training.stop("error")

    def training_start(self):
        if self.persona != "mentor":
            return False, "研修担当R1へ切り替えてください"
        if not self.auto_motion:
            return False, "会話の身ぶりをオンにしてから研修を開始してください"
        return self.submit("説明と見本", self._run_training, "研修を始めて", "研修開始ボタン")

    def training_stop(self):
        with self.control_lock:
            self.hold_epoch += 1
            self.training.stop()
            state = self.body.state()
            if self.body_enabled and (state["busy"] or state["stop_required"]):
                self.body.stop()
        self.add("system", "研修を中断しました。次の実演は開始しません")
        return True, "ok"

    def _say(self, text: str) -> None:
        self.add("r1", text, gesture="", direct=True)
        log(f"R1（そのまま） > {text}")
        self._speak(text)

    def _greet(self) -> None:
        self._chat("自己紹介して", "あいさつボタン")

    # ------------------------------------------------------------ 入口
    def chat(self, text: str):
        if "stop_move" in requested_body_actions(text):
            return self.body_stop()
        return self.submit("考え中", self._chat, text, "文字")

    def body_move(self, action: str):
        if action == "stop_move":
            return self.body_stop()
        if action not in BODY_DIRECTIONS:
            return False, "不明な移動です"
        if not self.body_enabled:
            return False, "歩行制御は有効になっていません"
        if not self.args.local and (not self.fsm_ok or self.fsm != 811):
            return False, "Running(811)の状態を確認してから操作してください"

        def run():
            if self.gesturer.thread is not None:
                self.add("system", "腕の動作が終了していないため移動しません")
                return
            ok, message = self.body.move(BODY_DIRECTIONS[action], self._body_duration(action))
            log(f"[歩行] {message}")
            if not ok:
                self.add("system", message)
                return
            self.body.wait_until_idle()
            result = self.body.state()
            if result["error"]:
                self.add("system", result["error"])
        return self.submit("移動中", run)

    def body_hold_start(self, action: str, token, epoch):
        if not isinstance(action, str) or action not in BODY_DIRECTIONS or action == "stop_move":
            return False, "不明な移動方向です"
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            return False, "画面の状態を更新してから操作してください"
        with self.control_lock:
            if epoch != self.hold_epoch:
                return False, "停止前の古い押下依頼は実行しません"
            if self.closed or not self.body_enabled:
                return False, "歩行制御は有効になっていません"
            if not self.args.local and (not self.fsm_ok or self.fsm != 811):
                return False, "Running(811)の状態を確認してから操作してください"
            if self.rec_stream is not None or not self.job_lock.acquire(blocking=False):
                return False, "録音・会話・別の動作が終わってから操作してください"
            self.busy = "押下歩行中"
            try:
                ok, message = self.body.start_hold(BODY_DIRECTIONS[action], token,
                                                    linear_speed=self.hold_linear_speed,
                                                    yaw_speed=self.hold_yaw_speed)
            except Exception:
                self.busy = ""
                self.job_lock.release()
                raise
            if not ok:
                self.busy = ""
                self.job_lock.release()
                return False, message
            self.training.stop()
            self._hold_token = token

        def release_job():
            with self.control_lock:
                if self._hold_token == token:
                    self._hold_token = None
                    self.busy = ""
                    self.job_lock.release()

        def watch():
            try:
                self.body.wait_until_idle()
                result = self.body.state()
                rc.log_turn(mode="manual_hold", connection_kind=self.connection_kind, body=result)
                if result.get("error"):
                    self.add("system", result["error"])
            finally:
                release_job()
        try:
            threading.Thread(target=watch, daemon=True, name="r1-hold-job").start()
        except Exception:
            try:
                self.body.release_hold(token)
                self.body.wait_until_idle()
            finally:
                release_job()
            raise
        return True, message

    def body_hold_heartbeat(self, token):
        return self.body.heartbeat_hold(token)

    def body_hold_release(self, token):
        # A release can arrive before its start request; the controller retires the token.
        return self.body.release_hold(token)

    def _body_duration(self, action: str) -> float:
        return self.forward_duration if action == "move_forward" else 1.0

    def body_stop(self):
        with self.control_lock:
            self.hold_epoch += 1
            self.training.stop()
            self._cancel_stand()
        if not self.body_enabled:
            return False, "歩行制御は有効になっていません"
        ok, message = self.body.stop()
        log(f"[歩行] {message}")
        self.add("system", message)
        return ok, message

    def stand_up(self, posture: str, ready: bool):
        if posture != "supine" or ready is not True:
            return False, "現在の姿勢が仰向けであることと、起立の準備を確認してください"
        if not self.body_enabled:
            return False, "全身制御を有効にして起動してください（--body-control）"
        with self.control_lock:
            return self.submit("起立中", self._run_stand, posture, ready, None, self.posture_epoch)

    def _run_stand(self, posture: str, ready: bool, should_continue=None, epoch=None):
        epoch = self.posture_epoch if epoch is None else epoch
        current = lambda: (not self.closed and epoch == self.posture_epoch
                           and (should_continue is None or should_continue()))
        if not current():
            return {"status": "cancelled", "error": "起立の依頼は取り消されています"}
        if not self.body_enabled:
            self.add("system", "全身制御が無効のため起立指令は送りません")
            return {"error": "body_disabled"}
        self.training.stop()
        result = self.posture.run(posture, ready, should_continue=current)
        status = result.get("status")
        if status == "preview":
            message = "起立手順のプレビューです。実機への指令は送っていません"
        elif status in ("finished", "already_running"):
            message = "Running（811）を読み取りました。姿勢は目視でも確認してください"
        elif status == "cancelled":
            message = "起立の後続手順を中断しました。開始済みの機体動作の即時停止は保証しません"
        else:
            message = "起立手順を中断しました: " + str(result.get("error") or status)
        self.add("system", message)
        rc.log_turn(mode="posture", connection_kind=self.connection_kind, result=result)
        return result

    def stand_cancel(self):
        self._cancel_stand()
        self.add("system", "起立の後続手順を中断します。緊急停止はコントローラーで行ってください")
        return True, "ok"

    def _cancel_stand(self):
        with self.control_lock:
            self.posture_epoch += 1
        self.posture.cancel()

    def say(self, text: str):
        return self.submit("発話中", self._say, text)

    def greet(self):
        return self.submit("あいさつ中", self._greet)

    def gesture(self, key: str) -> tuple[bool, str]:
        if key not in BUTTON_IDS:
            return False, "不明な動作です"
        aid, label = BUTTON_IDS[key]
        if not self.args.local and aid != 99 and (not self.fsm_ok or self.fsm not in rc.STANDING_FSM):
            return False, f"立っていない（モード {self.fsm}）ので送りません"
        def run():
            code = self.gesturer.perform(key) if aid is None else self.gesturer.manual(aid)
            log(f"[動作] {label}（ID {aid}） → {self.gesturer.describe(code)}")
            if code != 0:
                self.add("system", f"{label}: {self.gesturer.describe(code)}")
        return self.submit("動作中", run)

    def set_persona(self, key: str):
        key = personas.normalize_persona_key(key)
        if key not in personas.PERSONAS:
            return False, "不明な人格です"
        if not self.args.allow_persona_switch and key != self.persona:
            return False, "新人研修では人格を固定しています"
        with self.control_lock:
            if self.job_lock.locked() or self.rec_stream is not None or self._call_active():
                return False, "通話・録音・処理が終わってから切り替えてください"
            if key == "r1":
                self._openai_brain()
            self.persona = key
            self.training.stop("idle")
            label = self._persona_label(key)
            self.add("system", f"人格を「{label}」に切り替えました")
            log(f"[人格] {label} に切り替え")
            return True, "ok"

    def reset(self):
        with self.control_lock:
            if self.job_lock.locked() or self.rec_stream is not None or self._call_active():
                return False, "通話・録音・処理が終わってからリセットしてください"
            return self._reset_idle()

    def _reset_idle(self):
        self.training.stop("idle")
        brain = self.brain
        if hasattr(brain, "reset"):
            brain.reset()
        else:
            brain.previous_id = None
        self.add("system", "会話をリセットしました。人格と研修資料は保持しています")
        log("[会話] リセット")
        return True, "ok"

    def set_motion(self, on: bool):
        self.auto_motion = bool(on)
        log(f"[身ぶり] 会話の身ぶりを{'オン' if on else 'オフ'}にしました")
        return True, "ok"

    def record_start(self):
        with self.control_lock:
            return self._record_start()

    def _record_start(self):
        import sounddevice as sd
        if not self.stt:
            return False, "音声認識なし（--no-stt）で起動しています"
        if self._call_active():
            return False, "通話中です。録音ボタンを使うときは通話を終了してください"
        if self.call and self.call.state().get("input_closing"):
            return False, "前のマイク入力を終了中です。少し待ってから録音してください"
        if self.rec_stream is not None:
            return False, "録音中です"
        if self.job_lock.locked():
            return False, f"いま「{self.busy}」の途中です"
        mic = self.args.mic_device
        mic = int(mic) if mic and str(mic).isdigit() else mic
        if mic is not None:
            mic, metadata = resolve_input_device(mic, sd)
            log("[マイク] " + json.dumps(metadata, ensure_ascii=False))
        self.rec_frames = []
        stream = sd.InputStream(samplerate=rc.SAMPLE_RATE, channels=1, dtype="int16", device=mic,
                                callback=lambda data, *_: self.rec_frames.append(data.copy()))
        try:
            stream.start()
        except Exception:
            stream.close()
            raise
        self.rec_stream, self.rec_started = stream, time.time()
        log("[録音] 開始")
        return True, "ok"

    def record_stop(self):
        with self.control_lock:
            return self._record_stop()

    def _record_stop(self):
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

    def _listen(self, audio: np.ndarray, should_continue=None) -> None:
        current = lambda: not self.closed and (should_continue is None or should_continue())
        if not current():
            return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            with wave.open(path, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(rc.SAMPLE_RATE)
                w.writeframes(audio.astype("<i2").tobytes())
            t0 = time.time()
            if hasattr(self.stt, "transcribe_detail"):
                detail = self.stt.transcribe_detail(path)
            else:   # 古い認識器・テスト用の偽物
                detail = {"text": self.stt.transcribe(path)}
        finally:
            os.unlink(path)
        stt_s = round(time.time() - t0, 2)
        if not current():
            return
        text = detail.get("text") or ""
        language = detail.get("language") if detail.get("language") in ("ja", "en") else rc.text_language(text)
        low = self._low_confidence(detail)
        info = f"{language}" + (f" {detail['language_prob']:.2f}" if isinstance(detail.get("language_prob"), (int, float)) else "")
        log(f"  認識 ({stt_s:.1f}s, {info}{'、確信度低め' if low else ''}): {text or '（聞き取れませんでした）'}")
        if not text:
            self.add("system", bilingual.NOT_HEARD[language])
            return
        self.busy = "考え中"
        record = {"stt_s": stt_s, "audio_s": round(len(audio) / rc.SAMPLE_RATE, 2),
                  **{f"stt_{k}": detail[k] for k in ("language_prob", "confidence", "avg_logprob") if detail.get(k) is not None}}
        self._chat(text, "声", record, should_continue=current, language=language, low_confidence=low)

    def _low_confidence(self, detail: dict) -> bool:
        """Whisper の平均対数確率が低い、または日英の判定が五分五分なら確信度が低いとみなす。"""
        logprob, lang_p = detail.get("avg_logprob"), detail.get("language_prob")
        limit = getattr(self.args, "stt_low_logprob", -1.0)
        if detail.get("retried_as"):   # なまりの強い英語として文字にし直した → 聞き間違いの可能性が高い
            return True
        return bool((isinstance(logprob, (int, float)) and logprob < limit)
                    or (isinstance(lang_p, (int, float)) and lang_p < 0.6))

    def _call_active(self):
        return bool(self.call and self.call.state()["active"])

    def _can_listen(self):
        return not self.closed and not self.job_lock.locked() and self.rec_stream is None

    def _call_error(self, message):
        log(f"[通話] {message}")
        self.add("system", f"通話: {message}")

    def _call_utterance(self, audio, is_current):
        with self.control_lock:
            if not is_current() or not self.job_lock.acquire(blocking=False):
                return
            self.busy = "聞き取り中"
        try:
            self._listen(audio, should_continue=is_current)
        finally:
            self.busy = ""
            self.job_lock.release()

    def call_start(self):
        with self.control_lock:
            if not self.stt:
                return False, "音声認識なしで起動しています"
            if self.job_lock.locked() or self.rec_stream is not None:
                return False, "録音・処理が終わってから通話を開始してください"
            if self._call_active():
                return True, "通話中です"
            self.call.start()
            log("[通話] 開始。話し終わると自動で送り、返事のあと再び聞き取ります")
            return True, "ok"

    def call_stop(self):
        with self.control_lock:
            self.hold_epoch += 1
            held_token = self._hold_token
            self._cancel_stand()
            if self.training.active:
                self.training_stop()
            if self.call:
                self.call.stop()
            log("[通話] 終了")
        if held_token is not None:
            self.body.release_hold(held_token)
        return True, "ok"

    def set_call_threshold(self, value):
        if isinstance(value, bool):
            return False, "しきい値には数値を指定してください"
        self.call.set_threshold(value)
        return True, "ok"

    def material_add(self, name, text):
        with self.control_lock:
            if self.job_lock.locked() or self.rec_stream is not None or self._call_active():
                return False, "通話・録音・処理が終わってから資料を変更してください"
            self.materials.add_text(name, text)
            self.training.stop("idle")
            self.brains["mentor"].set_training_context(self.materials.context())
            self.brains["mentor"].reset()
            self.add("system", f"研修資料「{name}」を読み込みました。前の会話はリセットしました")
            return True, "ok"

    def material_remove(self, document_id):
        with self.control_lock:
            if self.job_lock.locked() or self.rec_stream is not None or self._call_active():
                return False, "通話・録音・処理が終わってから資料を変更してください"
            self.materials.remove(document_id)
            self.training.stop("idle")
            self.brains["mentor"].set_training_context(self.materials.context())
            self.brains["mentor"].reset()
            self.add("system", "研修資料を削除し、会話をリセットしました")
            return True, "ok"

    def close(self):
        self.closed = True
        self._cancel_stand()
        self.training.stop()
        self.call_stop()
        if self.body_enabled and (self.body.state()["busy"] or self.body.state()["stop_required"]):
            self.body.stop()
            self.body.wait_until_idle(4.0)
        if self.body_feedback:
            self.body_feedback.close()
        if self.rec_stream:
            self.rec_stream.close()
            self.rec_stream = None
        if self.stt:
            self.stt.close()
        for brain in {id(b): b for b in [*self.brains.values(), *self.mentor_backends.values()]}.values():
            if getattr(brain, "relay", None):
                brain.relay.close()
        if self.mentor_openai_client:
            self.mentor_openai_client.close()
        for tts in {id(t): t for t in (self.tts, self.tts_en) if t is not None}.values():
            proc = getattr(tts, "proc", None)
            if proc and proc.poll() is None:
                proc.terminate()


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
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            if host not in allowed or (origin and origin not in {f"http://{h}" for h in allowed}):
                return self._json({"ok": False, "message": "このPCのパネルから操作してください"}, 403)
            if self.headers.get_content_type() != "application/json":
                return self._json({"ok": False, "message": "JSON形式で送信してください"}, 415)
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if not 0 < length <= 131072:
                    return self._json({"ok": False, "message": "送信内容が空か、大きすぎます"}, 413)
                data = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(data, dict):
                    raise ValueError("JSONオブジェクトが必要です")
            except (ValueError, UnicodeError):
                return self._json({"ok": False, "message": "送信内容を読み取れません"}, 400)
            text = str(data.get("text", "")).strip()[:300]
            routes = {
                "/api/chat": lambda: engine.chat(text) if text else (False, "文が空です"),
                "/api/say": lambda: engine.say(text) if text else (False, "文が空です"),
                "/api/greet": engine.greet,
                "/api/training/start": engine.training_start,
                "/api/training/stop": engine.training_stop,
                "/api/gesture": lambda: engine.gesture(str(data.get("key", ""))),
                "/api/body/move": lambda: engine.body_move(str(data.get("action", ""))),
                "/api/body/hold/start": lambda: engine.body_hold_start(data.get("action"), data.get("token"), data.get("epoch")),
                "/api/body/hold/heartbeat": lambda: engine.body_hold_heartbeat(data.get("token")),
                "/api/body/hold/release": lambda: engine.body_hold_release(data.get("token")),
                "/api/body/stop": engine.body_stop,
                "/api/posture/stand": lambda: engine.stand_up(data.get("posture"), data.get("ready")),
                "/api/posture/cancel": engine.stand_cancel,
                "/api/record/start": engine.record_start,
                "/api/record/stop": engine.record_stop,
                "/api/reset": engine.reset,
                "/api/motion": lambda: engine.set_motion(data.get("on")),
                "/api/persona": lambda: engine.set_persona(str(data.get("key", ""))),
                "/api/mentor/backend": lambda: engine.set_mentor_backend(str(data.get("key", ""))),
                "/api/call/start": engine.call_start,
                "/api/call/stop": engine.call_stop,
                "/api/call/threshold": lambda: engine.set_call_threshold(data.get("value")),
                "/api/materials/add": lambda: engine.material_add(str(data.get("name", "")),
                                                                    str(data.get("text", ""))),
                "/api/materials/remove": lambda: engine.material_remove(str(data.get("id", ""))),
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
    p.add_argument("--dds-address", default=os.environ.get("R1_DDS_ADDRESS", ""),
                   help="DDSが使うこのPCのIPv4。アダプターに複数のIPがある場合に明示する")
    p.add_argument("--local", action="store_true", help="R1 を使わず PC のスピーカーで鳴らす")
    p.add_argument("--motion", action=argparse.BooleanOptionalAction, default=True,
                   help="会話ごとにStrataが身ぶりを選ぶ（既定オン、--no-motionでオフ）")
    p.add_argument("--body-control", action="store_true", help="実機で短時間の歩行・旋回を有効にする（ローカルは常にプレビュー可）")
    p.add_argument("--require-stop-verification", action="store_true",
                   help="実機の歩行をロックして起動し、ゼロ速度後の新しい静止確認で解除する")
    p.add_argument("--velocity-127-compatible", action="store_true", help="7105の応答127に互換対応する。実際の動作・停止確認は別途必要")
    p.add_argument("--forward-speed", type=float, default=0.1,
                   help="前進だけの速度（0より大きく0.3m/s以下、既定0.1）")
    p.add_argument("--forward-duration", type=float, default=1.0,
                   help="前進だけの時間（0より大きく2秒以下、既定1）")
    p.add_argument("--hold-linear-speed", type=float, default=0.9,
                   help="押している間の手動歩行速度（0より大きく0.9m/s以下）")
    p.add_argument("--hold-yaw-speed", type=float, default=1.8,
                   help="押している間の手動旋回速度（0より大きく1.8rad/s以下）")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--volume", type=int, default=100)
    p.add_argument("--gain", type=float, default=12.0412,
                   help="送信前PCMの増幅量(dB)。既定は元の約4倍のソフトリミッター付き増幅")
    p.add_argument("--voice", default="Microsoft Ichiro")
    p.add_argument("--voice-en", default="Microsoft David", help="英語の返事を読む Windows の声（男性: David / Mark）")
    p.add_argument("--rate", type=int, default=0)
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5-mini"))
    p.add_argument("--reasoning-effort", default="minimal")
    p.add_argument("--persona", choices=[*personas.PERSONAS, "omoshiro"], default="mentor", help="起動時の人格")
    p.add_argument("--allow-persona-switch", action="store_true", help="パネルの人格切り替えを許可する（既定は固定）")
    p.add_argument("--mentor-backend", choices=("strata", "openai"), default="strata",
                   help="研修担当R1の応答先。人格・教材は共通（GPT APIは従量課金）")
    p.add_argument("--strata-url", default=os.environ.get("R1_STRATA_URL", "http://127.0.0.1:8081/v1"),
                   help="Strataの /v1 のURL。Z840へは --z840-* によるSSHコマンド経由")
    p.add_argument("--strata-model", default="", help="Strata のモデル名（空なら /v1/models の先頭）")
    p.add_argument("--strata-timeout", type=float, default=10.0,
                   help="Strataの通信1回の待ち時間（1〜120秒、既定10。形式再試行・初回接続は別）")
    p.add_argument("--z840-host", default=os.environ.get("R1_Z840_HOST", ""), help="Z840 の SSH 先（Tailscale の IP など）")
    p.add_argument("--z840-user", default=os.environ.get("R1_Z840_USER", ""))
    p.add_argument("--z840-key-dir", default=os.environ.get("R1_Z840_KEY_DIR", ""), help="id_ed25519 と known_hosts のフォルダ")
    p.add_argument("--z840-hostkey-alias", default=os.environ.get("R1_Z840_HOSTKEY_ALIAS", ""))
    p.add_argument("--no-stt", action="store_true", help="音声認識を読み込まない（文字だけ）")
    p.add_argument("--stt-python", default=os.environ.get("R1_STT_PYTHON", str(rc.DEFAULT_STT_PYTHON)))
    p.add_argument("--stt-model", default="small")
    p.add_argument("--stt-device", default="cuda")
    p.add_argument("--stt-compute-type", default="float16")
    p.add_argument("--stt-language", default="auto",
                   help="音声認識の言語: auto（日本語と英語を聞き分け、日本語優先。既定）/ ja / en")
    p.add_argument("--stt-low-logprob", type=float, default=-1.0,
                   help="この値より認識の平均対数確率が低いと、機械操作の内容は聞き返させる（既定 -1.0）")
    p.add_argument("--cuda-dll-dir", default=rc.DEFAULT_CUDA_DLL_DIR)
    p.add_argument("--mic-device", default=None)
    p.add_argument("--call-threshold", type=float, default=0.012, help="通話の発話判定しきい値（0より大きく1以下）")
    p.add_argument("--call-silence", type=float, default=0.8, help="通話で発話の終了とみなす無音秒数")
    p.add_argument("--materials-file", default=str(rc.REPO_ROOT / "logs" / "training" / "materials.json"),
                   help="研修資料の保存先（既定はGit対象外）")
    p.add_argument("--hat-action", default="hiyari_hat_v1", help="実機で教示・登録した帽子の動作名")
    p.add_argument("--hat-fallback", action="store_true", help="帽子の教示動作がない場合に、頭上の手振りで代用する")
    p.add_argument("--log-dir", default=str(rc.REPO_ROOT / "logs"))
    args = p.parse_args()
    try:
        _forward_settings(args)
        _strata_timeout(args)
    except ValueError as error:
        p.error(str(error))
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
        engine.close()


if __name__ == "__main__":
    main()
