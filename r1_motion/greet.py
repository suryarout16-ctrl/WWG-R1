#!/usr/bin/env python3
"""あいさつデモ: 脱力(Zero Torque)を待つ → 直立 → 起立(Running) → 手を振りながらあいさつ → 握手 → 手を戻す

各段で人が確認してから進む（y を入力）。オペレーターは E-Stop を持ってそばにいること。

  pixi run python r1_motion/greet.py -i Ethernet                 # 本番
  pixi run python r1_motion/greet.py -i Ethernet --skip-stand    # すでに Running なら起立を飛ばす

起立の方法（運用ガイドのコントローラー手順と同じ順番）:
  寝ている: 0 → 1(Damping) → 4(Locked Stand) → 701(起き上がり) → 自動で 811(Running)
  吊っている: 0 → 1 → 4 → [接地させて肩を支える] → 811(Start)
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "r1_chat"))

import stand                                   # noqa: E402  (R1Loco, ログ, 確認入力)
from stand import R1Loco, log, name            # noqa: E402
from arm_action import ACTIONS, connect as connect_arm, describe   # noqa: E402

AUTO = False   # --auto: 確認入力を省いて進める（現場で安全を確認済みのときだけ）


def confirm(prompt: str, word: str = "y") -> bool:
    if AUTO:
        log(f"{prompt} → 自動で実行（--auto）")
        return True
    return stand.confirm(prompt, word)


class LockedLoco(R1Loco):
    """命令と監視を 1 つの接続で順番に送る（同時に呼ぶと応答待ちが 3104 で切れることがある）。"""

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()

    def _Call(self, api_id, parameter):
        with self._lock:
            return super()._Call(api_id, parameter)

    def read_fsm(self, tries: int = 5):
        """読めなければ少し待って再試行。最後まで読めなければ None。"""
        for i in range(tries):
            try:
                return self.get_fsm()
            except Exception as error:
                log(f"    モード読み出し失敗 ({i + 1}/{tries}): {error}")
                time.sleep(0.5)
        return None

GREETING = "こんにちは！僕はR1です。今日は来てくれてありがとう。よろしくお願いします。"
SHAKE_LINE = "握手しましょう。"
BYE_LINE = "ありがとう。楽しんでいってくださいね。"


class Voice:
    """r1_chat と同じ方法（Windows の Ichiro → PlayStream）で R1 に日本語を話させる。"""

    def __init__(self, volume: int, gain: float):
        from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
        import r1_chat
        self.r1_chat, self.gain = r1_chat, gain
        self.tts = r1_chat.SapiTTS("Microsoft Ichiro", 0)
        self.client = AudioClient()
        self.client.SetTimeout(10.0)
        self.client.Init()
        time.sleep(0.5)
        log(f"[声] SetVolume({volume}) → code={self.client.SetVolume(volume)}")

    def say(self, text: str, wait: bool = True) -> float:
        pcm = self.r1_chat.boost(self.tts.synth(text), self.gain)
        sid = uuid.uuid4().hex
        for i in range(0, len(pcm), self.r1_chat.CHUNK_BYTES):
            code, _ = self.client.PlayStream("r1_greet", sid, pcm[i:i + self.r1_chat.CHUNK_BYTES])
            if code != 0:
                log(f"[声] PlayStream code={code}")
                break
        seconds = len(pcm) / 32000
        log(f"[声] 「{text}」 ({seconds:.1f}s)")
        if wait:
            time.sleep(seconds + 0.3)
            self.client.PlayStop("r1_greet")
        return seconds


def watch_fsm(client: R1Loco, stop: threading.Event) -> None:
    """モードが変わるたびにログに残す（監視用）。"""
    last = None
    while not stop.is_set():
        try:
            fsm = client.read_fsm(1)
            if fsm is not None and fsm != last:
                log(f"[監視] モード → {name(fsm)}")
                last = fsm
        except Exception as error:   # 読めないことも記録
            log(f"[監視] 読み出し失敗: {error}")
        stop.wait(0.5)


def set_and_wait(client: R1Loco, target: int, expect: int, timeout: float) -> bool:
    code = client.set_fsm(target)
    log(f"    SetFsmId({target}) → code={code}")
    if code != 0:
        return False
    end = time.time() + timeout
    while time.time() < end:
        if client.read_fsm() == expect:
            return True
        time.sleep(0.3)
    log(f"    {timeout:.0f}秒待っても {name(expect)} になりませんでした（現在 {name(client.read_fsm())}）")
    return False


def stand_up(client: R1Loco, posture: str | None) -> bool:
    log("")
    log("=== 起立 ===")
    if posture:
        log(f"姿勢: {'仰向けに寝ている' if posture == 'l' else '吊っている'}（--posture {posture}）")
    else:
        posture = input("R1 の今の姿勢は？ [l=仰向けに寝ている / h=吊っている / それ以外=中止] > ").strip().lower()
        stand._log.write(f"{time.strftime('%H:%M:%S')} 入力: {posture!r}\n")
    if posture not in ("l", "h"):
        log("中止しました")
        return False
    log("確認: E-Stop を持ったオペレーターがいる / 周囲 2m に人・物がない / スタッフの許可あり")
    if not confirm("すべて満たしていますか？", "OK"):
        log("中止しました（何も送っていません）")
        return False

    fsm = client.read_fsm()
    if fsm == 0:
        if not confirm("① Damping（ブレーキ、体は動かない）にしますか？"):
            return False
        if not set_and_wait(client, 1, 1, 6):
            return False
    if client.read_fsm() == 1:
        if not confirm("② Locked Stand（関節が直立姿勢へ動く）にしますか？"):
            return False
        if not set_and_wait(client, 4, 4, 8):
            return False
        log("    関節が直立姿勢に入るまで 3 秒待ちます")
        time.sleep(3.0)

    if posture == "l":
        if not confirm("③ 起き上がり（701：その場で立ち上がる。周りを空けて）を始めますか？"):
            return False
        if not set_and_wait(client, 701, 811, 25):
            return False
    else:
        log("③ R1 を接地させ、両足が床に着いたら、1 人が肩をしっかり支えてください。")
        if not confirm("接地して肩を支えていますか？ Running（811：自分でバランスを取る）にします"):
            return False
        if not set_and_wait(client, 811, 811, 8):
            return False
    log("起立完了: Running（811）。バランスが落ち着くまで 4 秒待ちます")
    time.sleep(4.0)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="R1 あいさつデモ（脱力→直立→手振り・あいさつ→握手）")
    p.add_argument("-i", "--interface", required=True)
    p.add_argument("--skip-stand", action="store_true", help="すでに Running(811) のとき、起立を飛ばす")
    p.add_argument("--volume", type=int, default=100)
    p.add_argument("--gain", type=float, default=9.0)
    p.add_argument("--posture", choices=["l", "h"], help="l=仰向けに寝ている h=吊っている（指定すると質問しない）")
    p.add_argument("--auto", action="store_true", help="各段の確認入力を省く（現場で安全確認済み・E-Stop 担当がいるときだけ）")
    p.add_argument("--shake-seconds", type=float, default=6.0, help="握手の手を出しておく秒数（最大18）")
    p.add_argument("--wave-id", type=int, default=25, choices=[25, 26], help="25=胸の前で手を振る 26=頭の上で手を振る")
    args = p.parse_args()
    args.shake_seconds = min(max(args.shake_seconds, 1.0), 18.0)
    global AUTO
    AUTO = args.auto
    if args.auto and not args.skip_stand and not args.posture:
        sys.exit("--auto のときは --posture l か h も指定してください")
    if not args.interface.isascii():
        sys.exit("-i には英数字のアダプタ名を指定してください（例: Ethernet）")

    stand.LOG_DIR.mkdir(exist_ok=True)
    log_path = stand.LOG_DIR / f"r1_motion_{time.strftime('%Y%m%d_%H%M%S')}_greet.log"
    stand._log = open(log_path, "a", encoding="utf-8", buffering=1)
    log(f"[log] {log_path}")

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    Path("/tmp").mkdir(exist_ok=True)
    ChannelFactoryInitialize(0, args.interface)
    client = LockedLoco()
    client.SetTimeout(5.0)
    client.Init()
    time.sleep(1.0)
    voice = Voice(args.volume, args.gain)
    arm = connect_arm()
    arm_held = False

    stop = threading.Event()
    threading.Thread(target=watch_fsm, args=(client, stop), daemon=True).start()
    try:
        fsm = client.read_fsm()
        if args.skip_stand:
            if fsm != 811:
                log(f"--skip-stand ですが今は {name(fsm)} です。Running(811) にしてから実行してください。")
                return
        else:
            if fsm != 0:
                log(f"今は {name(fsm)} です。脱力（Zero Torque, 紫）になるのを待ちます…（オペレーターがコントローラーで: 寝かせる L2+X 長押し → L2+Y）")
            while client.read_fsm() != 0:
                time.sleep(0.5)
            log("脱力（Zero Torque）になりました。")
            time.sleep(2.0)   # 切り替え直後は少し待つ
            if not stand_up(client, args.posture):
                return

        log("")
        log("=== あいさつ ===")
        if not confirm(f"④ 手を振りながらあいさつしますか？（{ACTIONS[args.wave_id]}、腕が動きます）"):
            return
        started = time.time()
        speech = voice.say(GREETING, wait=False)          # 話しながら手を振る
        code = arm.execute(args.wave_id)                  # 動作が終わるまで戻らない
        log(f"    ExecuteAction({args.wave_id} {ACTIONS[args.wave_id]}) → {describe(code)}")
        time.sleep(max(0.0, speech + 0.3 - (time.time() - started)))
        voice.client.PlayStop("r1_greet")

        if not confirm("⑤ 握手の手を出しますか？（右手が前に出ます）"):
            return
        voice.say(SHAKE_LINE, wait=False)
        code = arm.execute(27)
        log(f"    ExecuteAction(27 握手) → {describe(code)}")
        if code == 0:
            arm_held = True
            log(f"    {args.shake_seconds:.0f}秒そのまま（来場者と握手。保持は最大20秒で自動解除）")
            time.sleep(args.shake_seconds)
        code = arm.release()
        arm_held = False
        log(f"    ExecuteAction(99 腕を戻す) → {describe(code)}")
        voice.say(BYE_LINE)
        log("")
        log("完了。R1 は Running（811）のままです。止めるときはオペレーターがコントローラーで。")
    except KeyboardInterrupt:
        log("\n中断しました。速度0を送り、以降は何も送りません。")
        log(f"    StopMove → code={client.stop_move()}")
        if arm_held:
            log(f"    腕を戻す(99) → {describe(arm.release())}")
    except Exception:
        import traceback
        log("エラーで止まりました（以降は何も送りません）:\n" + traceback.format_exc())
    finally:
        time.sleep(0.6)
        stop.set()


if __name__ == "__main__":
    main()
