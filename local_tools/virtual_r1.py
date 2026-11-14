"""仮想 R1：R1 が無くても、パネル・会話・あいさつデモ・歩行の流れを PC だけで試すためのもの。

  - R1 と同じ名前の DDS サービスに応える: sport / arm / voice / motion_switcher / robot_state
  - 実機で録画した状態データ（logs/r1_snapshot_*）をくり返し流す: rt/lowstate、IMU、バッテリーなど
  - 声（PlayStream）は PC のスピーカーで鳴らす（--mute で無音）
  - モード（FSM）と腕の動作・保持は実機で分かった決まりをまねる。秒数などは目安

使い方（リポジトリのルート、別々のターミナルで）:
  pixi run python local_tools/virtual_r1.py                       # 既定はループバック（この PC の中だけ）
  pixi run ui -i "Loopback Pseudo-Interface 1" --motion           # パネルは同じ -i でつなぐ
  pixi run python r1_motion/greet.py -i "Loopback Pseudo-Interface 1" --posture l

安全: 本物の R1 が同じネットワークに見えたら起動しない（同じ名前のサービスが二重になるため）。
これはシミュレーターではない。脚・腕の物理は再現しない（歩行は脚の角度を少し揺らすだけ）。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import queue
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import r1_types
from unitree_sdk2py.core.channel import ChannelFactory, ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.core.channel_name import ChannelType, GetServerChannelName
from unitree_sdk2py.idl.unitree_api.msg.dds_ import Request_, Response_, ResponseHeader_, ResponseStatus_
from unitree_sdk2py.utils.crc import CRC

LOOPBACK = "Loopback Pseudo-Interface 1"

# 実機で確認した値
FSM_RUNNING, FSM_ARM = 811, 816
HOLD_IDS = {18, 27, 29, 33, 35}          # 姿勢を保つ動作（実機で 7401 になったもの＋資料）
HOLD_MAX_S = 20.0
ACTION_S = 4.0                           # 腕の動作が終わるまでの目安（実機ログ: 手振り 25 で約 4〜5 秒）
HOLD_REACH_S = 2.5                       # 保持の姿勢に入るまでの目安
STAND_UP_S = 3.5                         # 701 → 811（実機ログ: 約 4 秒）
LIE_DOWN_S = 4.0                         # 702 → 1（目安）
VELOCITY_CODE = 127                      # この R1 の 7105 は動いても止まっても 127 を返す

# 録画を流さないトピック（命令・RPC・自分で出すもの）
SKIP_PREFIX = ("rt/api/", "rr/", "rq/", "rt/lowcmd", "rt/arm_sdk", "rt/arm/action/state", "ros_discovery_info",
               "rt/parameter_events")

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(time.strftime("%H:%M:%S ") + msg, flush=True)


class Service:
    """unitree の RPC サーバー。要求ごとにスレッドで処理する（腕の動作中でも一覧などに答えられる）。"""

    def __init__(self, name: str, handlers: dict, version: str = "1.0.0.0"):
        self.name, self.handlers, self.version = name, handlers, version
        f = ChannelFactory()
        self.tx = f.CreateSendChannel(GetServerChannelName(name, ChannelType.SEND), Response_)
        self.rx = f.CreateRecvChannel(GetServerChannelName(name, ChannelType.RECV), Request_, self._on_request, 32)

    def _on_request(self, req):
        threading.Thread(target=self._handle, args=(req,), daemon=True).start()

    def _handle(self, req):
        api = req.header.identity.api_id
        code, data, binary = 0, "", []
        try:
            if api == 1:                              # API バージョンの問い合わせ
                data = self.version
            elif api in self.handlers:
                code, data = self.handlers[api](req.parameter, bytes(req.binary))
            else:
                code = 3203                           # 未実装（実機と同じ番号）
        except Exception as error:
            code = 3202
            log(f"[{self.name}] {api} の処理で例外: {type(error).__name__}: {error}")
        if req.header.policy.noreply:
            return
        resp = Response_(ResponseHeader_(req.header.identity, ResponseStatus_(code)), data if code == 0 else "", binary)
        self.tx.Write(resp, 1.0)


class VirtualR1:
    def __init__(self, snapshot: Path, start_fsm: int, mute: bool, speed: float):
        self.lock = threading.Lock()
        self.fsm = start_fsm
        self.volume = 100
        self.mute = mute
        self.rpc = {(r["service"], r["api"]): r for r in json.loads((snapshot / "rpc.json").read_text("utf-8"))}
        self.actions = self.rpc.get(("arm", 7107), {}).get("data") or [[]]
        self.action_names = {a["id"]: a["name"] for a in self.actions[0]} if self.actions and self.actions[0] else {}
        self.arm_busy = False
        self.holding: int | None = None
        self.hold_until = 0.0
        self.velocity = (0.0, 0.0, 0.0)
        self.velocity_until = 0.0
        self.speed = speed
        self.audio_q: queue.Queue = queue.Queue()
        self.crc = CRC()

    # ---------- sport ----------
    def _set_fsm(self, new: int, reason: str = "") -> None:
        with self.lock:
            old, self.fsm = self.fsm, new
        if old != new:
            log(f"[モード] {old} → {new} {reason}")

    def sport_get_fsm(self, param, _):
        return 0, json.dumps({"data": self.fsm})

    def sport_set_fsm(self, param, _):
        target = int(json.loads(param or "{}").get("data"))
        cur = self.fsm
        if target == 701:
            if cur != 4:
                log(f"[注意] 701（起き上がり）は 4（Locked Stand）から。今は {cur}（実機の応答は不明なので受け付ける）")
            self._set_fsm(701, "起き上がり中")
            threading.Timer(STAND_UP_S / self.speed, lambda: self.fsm == 701 and self._set_fsm(FSM_RUNNING, "起立完了")).start()
        elif target == 702:
            self._set_fsm(702, "寝る動作中")
            threading.Timer(LIE_DOWN_S / self.speed, lambda: self.fsm == 702 and self._set_fsm(1, "寝た（Damping）")).start()
        else:
            if target == 0:
                self._release_arm("脱力")
            self._set_fsm(target)
        return 0, ""

    def sport_set_velocity(self, param, _):
        p = json.loads(param or "{}")
        v = tuple(float(x) for x in p.get("velocity", [0, 0, 0]))
        d = float(p.get("duration", 1.0))
        with self.lock:
            running = self.fsm in (FSM_RUNNING, FSM_ARM)
            self.velocity = v if running else (0.0, 0.0, 0.0)
            self.velocity_until = time.monotonic() + d
        log(f"[歩行] 速度 {v} × {d:.1f}s {'' if running else '（Running でないので動かない）'} → {VELOCITY_CODE}")
        return VELOCITY_CODE, ""

    # ---------- arm ----------
    def _release_arm(self, why: str) -> None:
        with self.lock:
            had = self.holding
            self.holding = None
        if had is not None:
            log(f"[腕] 保持 {had} を解除（{why}）")
            if self.fsm == FSM_ARM:
                self._set_fsm(FSM_RUNNING, "腕を戻した")

    def _arm_state(self) -> dict:
        with self.lock:
            hid = self.holding
        return {"holding": hid is not None, "id": hid or 0, "name": self.action_names.get(hid, "") if hid else ""}

    def arm_execute(self, param, _):
        p = json.loads(param or "{}")
        if "action_id" in p:
            aid = int(p["action_id"])
        elif "data" in p:
            aid = int(p["data"])
        else:
            return 3204, ""
        if aid == 99:
            self._release_arm("ID 99")
            return 0, ""
        if aid not in self.action_names:
            return 3204, ""
        with self.lock:
            if self.holding is not None:
                if aid == self.holding:
                    self.holding = None
                    log(f"[腕] 同じ ID {aid} で保持を解除")
                    if self.fsm == FSM_ARM:
                        self.fsm = FSM_RUNNING
                    return 0, ""
                log(f"[腕] ID {aid}: 保持中（{self.holding}）なので 7401")
                return 7401, ""
            if self.arm_busy:
                log(f"[腕] ID {aid}: 動作中なので 7400")
                return 7400, ""
            if self.fsm not in (FSM_RUNNING, FSM_ARM):
                log(f"[注意] Running でないのに腕の動作 {aid}（今は {self.fsm}。実機の応答は不明なので受け付ける）")
            self.arm_busy = True
            prev_fsm = self.fsm
        name = self.action_names.get(aid, "")
        log(f"[腕] ID {aid} {name} 開始")
        if prev_fsm == FSM_RUNNING:
            self._set_fsm(FSM_ARM, "腕の動作中")
        hold = aid in HOLD_IDS
        time.sleep((HOLD_REACH_S if hold else ACTION_S) / self.speed)   # 実機と同じく終わるまで返さない
        with self.lock:
            self.arm_busy = False
            if hold:
                self.holding = aid
                self.hold_until = time.monotonic() + HOLD_MAX_S / self.speed
        if hold:
            log(f"[腕] ID {aid} 保持中（最大 {HOLD_MAX_S:.0f} 秒、99 か同じ ID で解除）")
        else:
            log(f"[腕] ID {aid} 終了")
            if self.fsm == FSM_ARM:
                self._set_fsm(FSM_RUNNING, "腕の動作終了")
        return 0, ""

    def arm_list(self, param, _):
        return 0, json.dumps(self.actions, ensure_ascii=False)

    def arm_custom(self, param, _):
        name = json.loads(param or "{}").get("action_name", "")
        log(f"[腕] 教示動作 {name!r}（仮想では中身なし。{ACTION_S:.0f} 秒待つだけ）")
        time.sleep(ACTION_S / self.speed)
        return 0, ""

    def arm_stop_custom(self, param, _):
        log("[腕] 教示動作の停止")
        return 0, ""

    # ---------- voice ----------
    def voice_get_volume(self, param, _):
        return 0, json.dumps({"volume": self.volume})

    def voice_set_volume(self, param, _):
        self.volume = int(json.loads(param or "{}").get("volume", self.volume))
        log(f"[声] 音量 {self.volume}")
        return 0, ""

    def voice_play(self, param, binary):
        p = json.loads(param or "{}")
        self.audio_q.put((p.get("stream_id"), binary))
        return 0, ""

    def voice_stop(self, param, _):
        while not self.audio_q.empty():
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break
        return 0, ""

    def voice_tts(self, param, _):
        log(f"[声] TtsMaker（R1 の内蔵 TTS は日本語不可）: {param[:80]}")
        return 0, ""

    def voice_led(self, param, _):
        return 0, ""

    def _audio_loop(self):
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError:
            log("[声] sounddevice が無いので声は鳴らさない")
            self.mute = True
        stream, last_id, n = None, None, 0
        while True:
            sid, pcm = self.audio_q.get()
            if sid != last_id:
                if last_id is not None:
                    log(f"[声] 再生 {n / 32000:.1f} 秒")
                last_id, n = sid, 0
            n += len(pcm)
            if self.mute:
                time.sleep(len(pcm) / 32000)
                continue
            if stream is None:
                stream = sd.OutputStream(samplerate=16000, channels=1, dtype="int16")
                stream.start()
            data = np.frombuffer(pcm, dtype=np.int16) * (self.volume / 100.0)
            stream.write(data.astype(np.int16))

    # ---------- 状態の再生 ----------
    def _patch(self, topic: str, msg, t: float):
        if topic.endswith("sportmodestate") and hasattr(msg, "fsm_id"):
            msg.fsm_id = self.fsm
        elif topic == "rt/lowstate" and hasattr(msg, "motor_state"):
            vx, vy, w = self.velocity
            now = time.monotonic()
            speed = max(abs(vx), abs(vy), abs(w) * 0.5)
            if now < self.velocity_until and speed > 0.12:
                # 実機の目安: 0.1 m/s×1 秒では足が出ず、0.3 m/s×2 秒で脚の変化幅 約 0.26 rad・最大 約 3 rad/s
                amp, hz = min(0.15, 0.72 * (speed - 0.12)), 2.5
                ph = 2 * math.pi * hz * now
                for j, sign in ((0, 1), (3, -1), (6, -1), (9, 1)):       # 左右の股・膝を逆位相で
                    msg.motor_state[j].q += sign * amp * math.sin(ph)
                    msg.motor_state[j].dq += sign * amp * 2 * math.pi * hz * math.cos(ph)
                msg.crc = self.crc.Crc(msg)
        return msg


    def _replay(self, topic: str, typ, samples: list):
        pub = ChannelPublisher(topic, typ)
        pub.Init()
        t0 = samples[0][0]
        rel = [t - t0 for t, _ in samples]
        span = (rel[-1] + (rel[-1] / max(len(rel) - 1, 1))) if len(rel) > 1 else 1.0
        # くり返すときに tick が戻ると「古いデータ」として捨てられるので、周回ごとに先へずらす
        ticks = [getattr(m, "tick", None) for _, m in samples]
        tick_span = 0
        if all(isinstance(t, int) for t in ticks) and len(ticks) > 1:
            tick_span = ticks[-1] - ticks[0] + max(1, (ticks[-1] - ticks[0]) // (len(ticks) - 1))
        start, i, loop = time.monotonic(), 0, 0
        while True:
            due = start + loop * span + rel[i]
            now = time.monotonic()
            if due > now:
                time.sleep(min(due - now, 0.05))
                continue
            msg = samples[i][1]
            if tick_span or topic in ("rt/sportmodestate", "rt/lf/sportmodestate"):
                msg = copy.deepcopy(msg)
                if tick_span:
                    msg.tick = (msg.tick + loop * tick_span) % 2 ** 32
                msg = self._patch(topic, msg, now)
                if tick_span and hasattr(msg, "crc"):
                    c = self.crc.Crc(msg)          # CRC を計算できる型（LowState など）だけ付け直す
                    if c is not None:
                        msg.crc = c
            try:
                pub.Write(msg, 0.5)
            except Exception:
                pass
            i += 1
            if i >= len(samples):
                i, loop = 0, loop + 1

    def _arm_state_loop(self):
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        pub = ChannelPublisher("rt/arm/action/state", String_)
        pub.Init()
        while True:
            with self.lock:
                if self.holding is not None and time.monotonic() > self.hold_until:
                    log(f"[腕] 保持 {self.holding} が {HOLD_MAX_S:.0f} 秒たったので自動で戻す")
                    self.holding = None
                    if self.fsm == FSM_ARM:
                        self.fsm = FSM_RUNNING
            pub.Write(String_(json.dumps(self._arm_state())), 0.5)
            time.sleep(0.6)

    def start_replay(self, snapshot: Path) -> None:
        sdk = r1_types_sdk()
        dyn = r1_types.load(snapshot)
        disc = json.loads((snapshot / "discovery.json").read_text("utf-8"))
        type_names = {p["topic"]: p["type"] for p in disc["publications"].values()}
        started = []
        for raw in sorted((snapshot / "raw").glob("*.pkl")):
            topic = next((n for n in type_names if n.replace("/", "_") + ".pkl" == raw.name), None)
            if not topic or topic.startswith(SKIP_PREFIX):
                continue
            typ = sdk.get(type_names[topic]) or dyn.get(topic)   # CRC を計算できるよう SDK の型を優先
            if typ is None:
                continue
            samples = []
            for t, b in r1_types.read_raw(raw):
                try:
                    samples.append((t, typ.deserialize(b)))
                except Exception:
                    pass
            if not samples:
                continue
            threading.Thread(target=self._replay, args=(topic, typ, samples), daemon=True).start()
            started.append(f"{topic}({len(samples)})")
        log(f"[再生] {len(started)} トピック: {', '.join(started)}")

    def start_services(self) -> list:
        ms = self.rpc.get(("motion_switcher", 1001), {}).get("data", {"form": "0", "name": "ai"})
        sl = self.rpc.get(("robot_state", 1003), {}).get("data", [])
        j = lambda x: (lambda p, b: (0, json.dumps(x, ensure_ascii=False)))
        return [
            Service("sport", {7001: self.sport_get_fsm, 7101: self.sport_set_fsm, 7105: self.sport_set_velocity}),
            Service("arm", {7106: self.arm_execute, 7107: self.arm_list, 7108: self.arm_custom,
                            7113: self.arm_stop_custom}),
            Service("voice", {1001: self.voice_tts, 1003: self.voice_play, 1004: self.voice_stop,
                              1005: self.voice_get_volume, 1006: self.voice_set_volume, 1010: self.voice_led}),
            Service("motion_switcher", {1001: j(ms), 1005: j({"silent": False})}, "1.0.0.1"),
            Service("robot_state", {1003: j(sl)}, "1.0.0.1"),
        ]


def r1_types_sdk() -> dict:
    """SDK にある型（CRC の計算ができる）を DDS の型名で引く。"""
    return r1_types_sdk.cache if hasattr(r1_types_sdk, "cache") else _load_sdk()


def _load_sdk() -> dict:
    import importlib
    import pkgutil
    import unitree_sdk2py.idl as idl
    out = {}
    for mod in pkgutil.walk_packages(idl.__path__, idl.__name__ + "."):
        try:
            m = importlib.import_module(mod.name)
        except Exception:
            continue
        for v in vars(m).values():
            name = getattr(v, "__idl_typename__", None)
            if isinstance(v, type) and name:
                out[name.replace(".", "::")] = v
    r1_types_sdk.cache = out
    return out


def real_r1_visible(seconds: float = 2.5) -> bool:
    """本物の R1（rt/lowstate を出している相手）が見えるか。"""
    from cyclonedds.builtin import BuiltinDataReader, BuiltinTopicDcpsPublication
    dp = ChannelFactory()._ChannelFactory__participant
    reader = BuiltinDataReader(dp, BuiltinTopicDcpsPublication)
    time.sleep(seconds)
    return any(s.topic_name in ("rt/lowstate", "rt/sportmodestate") for s in reader.take(N=10000))


def main():
    ap = argparse.ArgumentParser(description="仮想 R1（R1 なしでパネル・会話・デモの流れを試す）")
    ap.add_argument("-i", "--interface", default=LOOPBACK, help=f"DDS のネットワーク（既定: {LOOPBACK}）")
    ap.add_argument("--snapshot", default="", help="logs/r1_snapshot_* のフォルダ（既定: いちばん新しいもの）")
    ap.add_argument("--fsm", type=int, default=0, help="最初のモード（既定 0 = 脱力。811 で立った状態から）")
    ap.add_argument("--mute", action="store_true", help="声を鳴らさない")
    ap.add_argument("--speed", type=float, default=1.0, help="腕・起立の時間を何倍速にするか（テスト用）")
    args = ap.parse_args()

    snaps = sorted((ROOT / "logs").glob("r1_snapshot_*"))
    snapshot = Path(args.snapshot) if args.snapshot else (snaps[-1] if snaps else None)
    if snapshot is None or not (snapshot / "types.pkl").exists():
        sys.exit("スナップショット（types.pkl 付き）が見つかりません。local_tools/r1_snapshot.py と r1_types.py dump を先に")

    Path("/tmp").mkdir(exist_ok=True)
    ChannelFactoryInitialize(0, args.interface)
    if real_r1_visible():
        sys.exit(f"本物の R1 が {args.interface} に見えるので起動しません（同じ名前のサービスが二重になる）")

    r1 = VirtualR1(snapshot, args.fsm, args.mute, max(args.speed, 0.1))
    services = r1.start_services()
    threading.Thread(target=r1._audio_loop, daemon=True).start()
    threading.Thread(target=r1._arm_state_loop, daemon=True).start()
    r1.start_replay(snapshot)
    log(f"=== 仮想 R1 起動: -i \"{args.interface}\"  モード {r1.fsm}  録画 {snapshot.name}  （Ctrl+C で終了） ===")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log("終了")
    del services


if __name__ == "__main__":
    main()
