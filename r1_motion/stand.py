#!/usr/bin/env python3
"""R1 を PC から「直立（Locked Stand Mode, FSM 4）」まで遷移させる。1段ずつ人が確認して進める。

運用ガイドの手順（コントローラー操作）を SDK の FSM 指令に置き換えたもの:
  Zero Torque(0, 紫) --L2+B--> Damping(1, オレンジ) --L2+↑--> Locked Stand(4, 青) = 直立姿勢（姿勢制御なし）
  この先の Running(811, 白) へは、このスクリプトでは進めない（コントローラーで行う）

前提（必ず）:
  - R1 は「仰向けに寝ている」か「スタンド／ガントリーに吊られている」。Locked Stand は自分で立っていられない
  - コントローラーを持ったオペレーターが別にいて、いつでも E-Stop（無ければ L2+B 5秒長押し）を押せる
  - 周囲 2m 以上に人や物がない。スタッフの許可を取っている

使い方（リポジトリのルートで）:
  pixi run python r1_motion/stand.py -i Ethernet --status    # 今のモードを読むだけ（何も動かさない）
  pixi run python r1_motion/stand.py -i Ethernet             # 直立まで（各段で y を入力して進む）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.rpc.client import Client

FSM_NAMES = {0: "Zero Torque（脱力・紫）", 1: "Damping（ブレーキ・オレンジ）", 4: "Locked Stand（直立・青）",
             811: "Running（自立・歩行・白）", 816: "Running＋腕の動作中（Arm Action 実行中に観測）", 701: "Lie→Stand（起き上がり中）", 702: "Stand→Lie（寝る動作中）"}
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_log = None


def log(message: str) -> None:
    print(message, flush=True)
    if _log:
        _log.write(f"{time.strftime('%H:%M:%S')} {message}\n")


class R1Loco(Client):
    """R1 の sport サービス（unitree_sdk2_python の r1/loco と同じ API。配布 SDK に r1 が入っていないため最小限を自前で持つ）。"""

    def __init__(self):
        super().__init__("sport", False)

    def Init(self):
        self._SetApiVerson("1.0.0.0")
        self._RegistApi(7001, 0)   # GetFsmId
        self._RegistApi(7002, 0)   # GetFsmMode
        self._RegistApi(7101, 0)   # SetFsmId
        self._RegistApi(7105, 0)   # SetVelocity
        # 手振り・握手は sport ではなく arm サービス（arm_action.py）。sport の 7106 はこの R1 では 3203

    def stop_move(self) -> int:
        code, _ = self._Call(7105, json.dumps({"velocity": [0.0, 0.0, 0.0], "duration": 1.0}))
        return code

    def get_fsm(self) -> int | None:
        code, data = self._Call(7001, "{}")
        if code != 0:
            raise RuntimeError(f"FSM の読み出しに失敗しました (code={code})")
        return int(json.loads(data)["data"])

    def get_mode(self):
        code, data = self._Call(7002, "{}")
        return json.loads(data).get("data") if code == 0 and data else None

    def set_fsm(self, fsm_id: int) -> int:
        code, _ = self._Call(7101, json.dumps({"data": fsm_id}))
        return code


def name(fsm) -> str:
    return f"{fsm} {FSM_NAMES.get(fsm, '（不明）')}"


def confirm(prompt: str, word: str = "y") -> bool:
    answer = input(f"{prompt} [{word} で実行 / それ以外で中止] > ").strip()
    if _log:
        _log.write(f"{time.strftime('%H:%M:%S')} 入力: {answer!r}\n")
    return answer == word


def wait_for(client: R1Loco, target: int, timeout: float = 6.0) -> int | None:
    end = time.time() + timeout
    fsm = None
    while time.time() < end:
        fsm = client.get_fsm()
        if fsm == target:
            return fsm
        time.sleep(0.3)
    return fsm


def step(client: R1Loco, target: int, what: str, check: str) -> bool:
    log("")
    log(f"--- 次: {name(target)} にする  ({what})")
    log(f"    コントローラー操作なら: {check}")
    if not confirm("    R1 の周りは安全で、オペレーターは E-Stop を押せますか？"):
        log("    中止しました（何も送っていません）")
        return False
    code = client.set_fsm(target)
    log(f"    SetFsmId({target}) → code={code}")
    if code != 0:
        log("    失敗しました。ここで止めます。")
        return False
    now = wait_for(client, target)
    log(f"    現在: {name(now)}")
    if now != target:
        log("    目標のモードになりませんでした。ここで止めます。LED の色とオペレーターの確認をしてください。")
        return False
    return True


def main() -> None:
    global _log
    p = argparse.ArgumentParser(description="R1 を直立（Locked Stand）まで遷移させる")
    p.add_argument("-i", "--interface", required=True, help="R1 につないだ有線LANアダプタ名（例: Ethernet）")
    p.add_argument("--status", action="store_true", help="今のモードを表示するだけ")
    args = p.parse_args()
    if not args.interface.isascii():
        sys.exit("-i には英数字のアダプタ名を指定してください（例: Ethernet）")

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"r1_motion_{time.strftime('%Y%m%d_%H%M%S')}_{'status' if args.status else 'stand'}.log"
    _log = open(log_path, "a", encoding="utf-8", buffering=1)
    log(f"[log] {log_path}")

    Path("/tmp").mkdir(exist_ok=True)
    ChannelFactoryInitialize(0, args.interface)
    client = R1Loco()
    client.SetTimeout(5.0)
    client.Init()
    time.sleep(1.0)

    fsm = client.get_fsm()
    log(f"現在のモード: {name(fsm)}   (FSM mode={client.get_mode()})")
    if args.status:
        return

    if fsm in (811, 701, 702):
        log("すでに立っている／動作中です。このスクリプトでは何もしません。")
        return
    if fsm not in (0, 1, 4):
        log("想定外のモードです。何もせずに終わります。スタッフに確認してください。")
        return
    if fsm == 4:
        log("すでに直立（Locked Stand）です。")
        return

    log("")
    log("=== 直立（Locked Stand）までの手順 ===")
    log("確認: R1 は仰向けに寝ているか、スタンド／ガントリーに吊られている。")
    log("確認: コントローラーを持ったオペレーターがいて、E-Stop（無ければ L2+B 5秒長押し）を押せる。")
    log("確認: 周囲 2m に人・物がない。スタッフの許可を取った。")
    if not confirm("上の3つをすべて満たしていますか？", "OK"):
        log("中止しました（何も送っていません）")
        return

    if fsm == 0 and not step(client, 1, "全モーターに軽くブレーキ。体は動かない", "L2 + B（「Dump Mode」と話し、LED がオレンジ）"):
        return
    if not step(client, 4, "関節が直立姿勢へ動く。姿勢制御はないので支えが必要", "L2 + ↑（LED が青）"):
        return

    log("")
    log("完了: 直立（Locked Stand）になりました。")
    log("このモードは自分でバランスを取りません。吊ったまま／寝かせたままにしてください。")
    log("歩ける状態（Running）へはオペレーターがコントローラーで: 寝ている → L2+X 長押し / 吊っている → 接地して肩を支え R2+A")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n中断しました。以降は何も送りません。必要ならオペレーターがコントローラーで操作してください。")
