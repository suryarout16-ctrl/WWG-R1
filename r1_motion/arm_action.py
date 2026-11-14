#!/usr/bin/env python3
"""R1 の腕のプリセット動作（Arm Action サービス）。

Unitree R1 Arm Action Interface（unitree_sdk2 example/r1/high_level/r1_arm_action_example.cpp）を Python にしたもの。
サービス名 "arm"（rt/api/arm/request・response）、API バージョン 1.0.0.0。

  pixi run python r1_motion/arm_action.py -i Ethernet --list      # 使える動作の一覧（動かさない）
  pixi run python r1_motion/arm_action.py -i Ethernet --id 25     # 胸の前で手を振る
  pixi run python r1_motion/arm_action.py -i Ethernet --id 99     # 腕を初期姿勢に戻す

注意: 腕が大きく動く。周りに人・物がないときだけ。デバッグモードでは使えない。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from unitree_sdk2py.rpc.client import Client

API_EXECUTE = 7106          # プリセット動作（終わるまで戻らない）
API_LIST = 7107
API_EXECUTE_CUSTOM = 7108   # 教示で作った動作（すぐ戻る）
API_STOP_CUSTOM = 7113

ACTIONS = {
    99: "腕を初期姿勢に戻す", 11: "両手で投げキス", 12: "左手で投げキス", 13: "右手で投げキス",
    15: "両腕を肩の高さに上げる", 17: "拍手", 18: "ハイタッチ", 19: "ハグ", 22: "腕を交差（お断り）",
    23: "右腕を肩の高さに上げる", 24: "ウルトラマン光線", 25: "胸の前で手を振る", 26: "頭の上で手を振る",
    27: "握手", 28: "ガッツポーズ（左）", 29: "ガッツポーズ（右）", 30: "ガッツポーズ（両手）",
    31: "右腕を前に伸ばす", 33: "右手を胸に当てる", 34: "両腕を上げて右寄り", 35: "強調のしぐさ", 36: "前に押す",
}
# 終わったあと姿勢を保つ動作（最大 20 秒。99 か同じ ID で解除。保持中に別の ID は 7401）
HOLDS = {15, 18, 19, 22, 23, 24, 27, 28, 29, 30, 31, 33, 34, 35, 36}

ERRORS = {
    7399: "サーバー側の予期しないエラー", 7400: "上半身が使用中（rt/arm_sdk を使うプログラムがある／動作中）",
    7401: "腕が姿勢保持中（99 か同じ ID で解除）", 7402: "この ID はこのファームウェアに無い（--list で確認）",
    7403: "動作ファイルが無い／壊れている", 7404: "今のモードではこの動作はできない",
    7405: "名前が使用済み", 7406: "バッテリー残量不足（6% 未満）", 7407: "モーター異常",
    3103: "API がサーバーに登録されていない", 3104: "応答なし（タイムアウト）", 3204: "パラメータが不正",
}


def describe(code: int) -> str:
    return "OK" if code == 0 else f"code={code} {ERRORS.get(code, '')}".rstrip()


class R1ArmAction(Client):
    def __init__(self):
        super().__init__("arm", False)

    def Init(self):
        self._SetApiVerson("1.0.0.0")
        for api in (API_EXECUTE, API_LIST, API_EXECUTE_CUSTOM, API_STOP_CUSTOM):
            self._RegistApi(api, 0)

    param_key = "action_id"   # R1 ドキュメントの書式。G1 の Python SDK は "data"

    def execute(self, action_id: int) -> int:
        """プリセット動作。動作が終わるまで戻らないので、タイムアウトは 10 秒以上に。"""
        code, _ = self._Call(API_EXECUTE, json.dumps({self.param_key: action_id}))
        if code == 3204 and self.param_key == "action_id":   # パラメータ不正 → G1 の書式で再送
            self.param_key = "data"
            code, _ = self._Call(API_EXECUTE, json.dumps({self.param_key: action_id}))
        return code

    def release(self) -> int:
        return self.execute(99)

    def action_list(self):
        code, data = self._Call(API_LIST, "{}")
        return code, (json.loads(data) if code == 0 and data else None)

    def execute_custom(self, name: str) -> int:
        code, _ = self._Call(API_EXECUTE_CUSTOM, json.dumps({"action_name": name}))
        return code

    def stop_custom(self) -> int:
        code, _ = self._Call(API_STOP_CUSTOM, "")
        return code


def connect(timeout: float = 12.0) -> R1ArmAction:
    arm = R1ArmAction()
    arm.SetTimeout(timeout)
    arm.Init()
    return arm


def main() -> None:
    p = argparse.ArgumentParser(description="R1 の腕のプリセット動作")
    p.add_argument("-i", "--interface", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="使える動作の一覧（動かさない）")
    g.add_argument("--id", type=int, help="プリセット動作の ID")
    g.add_argument("--name", help="教示で作った動作の名前")
    g.add_argument("--stop", action="store_true", help="教示動作を止める")
    args = p.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    Path("/tmp").mkdir(exist_ok=True)
    ChannelFactoryInitialize(0, args.interface)
    arm = connect()
    time.sleep(1.0)

    if args.list:
        code, data = arm.action_list()
        if code != 0:
            sys.exit(f"一覧の取得に失敗: {describe(code)}")
        presets, customs = (data + [[], []])[:2] if isinstance(data, list) else (data, [])
        print("プリセット動作:")
        for a in presets or []:
            extra = f"  fsm={a['fsm']}" if "fsm" in a else ""
            print(f"  {a.get('id'):>3}  {a.get('name')}  {ACTIONS.get(a.get('id'), '')}{extra}")
        print("教示動作:")
        for a in customs or []:
            print(f"  {a.get('name')}  ({a.get('time')} 秒)")
        return
    if args.id is not None:
        print(f"ExecuteAction({args.id} {ACTIONS.get(args.id, '')}) → {describe(arm.execute(args.id))}")
    elif args.name:
        print(f"ExecuteAction('{args.name}') → {describe(arm.execute_custom(args.name))}")
    else:
        print(f"StopCustomAction → {describe(arm.stop_custom())}")


if __name__ == "__main__":
    main()
