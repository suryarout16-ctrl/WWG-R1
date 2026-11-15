#!/usr/bin/env python3
"""8770 パネルと同じ GPT（研修担当R1）に人間側の発話を順に送り、会話の結果を確かめる。

実機・音声・パネルには何も送らない（返事の文と、選ばれた身ぶりを記録するだけ）。
GPT API を呼ぶので有料。APIキーはパネルと同じく環境変数か Windows の資格情報から読む。

  .pixi\\envs\\default\\python.exe local_tools\\chat_verify.py 会話.txt
  .pixi\\envs\\default\\python.exe local_tools\\chat_verify.py 会話.txt --connection local

会話ファイル: 1行 = 人間側の1発話。空行と # で始まる行は無視。
「---」の行で会話履歴と研修状態を消して別の会話として始める。
「研修を始めてください」はパネルと同じく 説明→見本→質問 の3段階を R1 が続けて話す。
結果は logs/chat_verify/ に .md と .json で保存する。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "r1_chat"))
sys.path.insert(0, str(REPO / "r1_motion"))

import r1_chat as rc  # noqa: E402
import bilingual  # noqa: E402
from training_materials import TrainingMaterials  # noqa: E402
from training_flow import TrainingFlow, is_training_start  # noqa: E402
from body_control import requested_body_actions  # noqa: E402

BODY_ACTIONS = {"move_forward", "move_backward", "move_left", "move_right",
                "turn_left", "turn_right", "stop_move"}


class Session:
    """web_ui.Engine._chat / _run_training / _training_actions の会話部分だけを写したもの。"""

    def __init__(self, brain, connection: str, fsm: int, body_enabled: bool):
        self.brain, self.connection, self.fsm, self.body_enabled = brain, connection, fsm, body_enabled
        self.training = TrainingFlow()

    def training_actions(self, phase: str) -> set[str]:
        if phase == "demonstrate" and self.connection == "local" and self.body_enabled:
            return {"none", "move_forward"}
        if phase in ("explain", "demonstrate"):
            return {"none", "point_forward", "emphasize"}
        return {"none", *(name for name in rc.GESTURES if name != "hat_tip")}

    def runtime_context(self, training_step: str | None) -> str:
        if self.connection == "local":
            mode = "R1未接続のローカルプレビュー。動作は実機へ送らない。"
        else:
            mode = f"実機R1接続モード。読取FSM={self.fsm}、状態読取成功=True。"
        mode += "現在の応答生成はGPT API。"
        mode += (("短時間の歩行・旋回を許可。" if self.body_enabled else "歩行の実行は無効。") +
                 "新規の低レベル関節指令は未検証のため保留。工場の機械を直接操作する機能はない。")
        if self.training.active:
            mode += self.training.context(training_step)
            if training_step:
                choices = self.training_actions(training_step)
                mode += "\nこの段階で選べるgestureは" + ", ".join(sorted(choices)) + "のみ。"
                if "move_forward" in choices:
                    mode += ("研修開始操作にひもづく仮想の見本歩行許可がある。"
                             "move_forwardで荷物の移動を歩行で表現し、発話は移動前の短い説明にする。")
                else:
                    mode += "この段階では歩行を追加しない。許可された腕の身ぶりで説明する。"
        return mode

    def turn(self, user_text: str, training_step: str | None = None) -> dict:
        language = rc.text_language(user_text)
        self.brain.set_runtime_context(self.runtime_context(training_step))
        model_input = self.training.request(training_step) if training_step else user_text
        kwargs = {"language": language, "low_confidence": False}
        if training_step:
            kwargs.update(allowed_gestures=self.training_actions(training_step),
                          max_chars=180 if language == "en" else 100)
        t0 = time.time()
        error = None
        try:
            say, gesture = self.brain.reply(model_input, **kwargs)
        except Exception as exc:   # 例外本文にキーを含め得るので短く
            error = type(exc).__name__ + ": " + str(exc)[:120]
            say, gesture = bilingual.ERROR_REPLY[language], "none"
        elapsed = round(time.time() - t0, 2)
        blocked = None
        if training_step and gesture not in self.training_actions(training_step):
            blocked, gesture = gesture, "none"
        if training_step:
            allowed = self.training_actions(training_step) & BODY_ACTIONS
        else:
            allowed = requested_body_actions(user_text) | bilingual.requested_body_actions_en(user_text)
        if gesture in BODY_ACTIONS:
            if not (gesture in allowed and self.body_enabled):
                result = "送らない（頼まれていない動作）"
            elif self.connection == "local":
                result = "表示のみ（R1未接続）"
            else:
                result = "実機へ送る"
        elif gesture and gesture != "none":
            result = "腕の身ぶり" + ("（R1未接続なので表示のみ）" if self.connection == "local" else "")
        else:
            result = "なし"
        reply = getattr(self.brain, "last_reply", None)
        replaced = None
        if isinstance(reply, dict) and (reply.get("model_say"), reply.get("model_gesture")) \
                != (reply.get("say"), reply.get("gesture")):
            replaced = {"model_say": reply.get("model_say"), "model_gesture": reply.get("model_gesture")}
        if self.training.active and training_step is None and not error:
            self.training.learner_replied()
        return {"user": None if training_step else user_text, "training_step": training_step,
                "phase_after": self.training.phase, "language": language, "say": say,
                "gesture": gesture, "gesture_result": result, "training_blocked": blocked,
                "seconds": elapsed, "replaced": replaced, "error": error}

    def handle(self, user_text: str) -> list[dict]:
        if is_training_start(user_text):
            self.training.start()
            rows = []
            for step in self.training.STEPS:
                self.training.enter(step)
                row = self.turn(user_text, training_step=step)
                if step == "explain":
                    row["user"] = user_text
                rows.append(row)
            self.training.waiting()
            return rows
        return [self.turn(user_text)]

    def reset(self):
        self.brain.reset()
        self.training = TrainingFlow()


def build_brain(model: str, effort: str, materials_file: Path, body_enabled: bool):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        from credentials import get_secret
        api_key = get_secret()
    if not api_key:
        raise SystemExit("OpenAI の APIキーが見つかりません（パネルと同じ資格情報を使います）")
    from openai import OpenAI
    from mentor_backend import OpenAIMentorBrain
    client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1", timeout=20.0, max_retries=0)
    brain = OpenAIMentorBrain(client, model, effort)
    # 教材ファイルを書き換えないよう、一時コピーから読む
    tmp = Path(tempfile.mkdtemp(prefix="chat_verify_")) / materials_file.name
    if materials_file.exists():
        shutil.copyfile(materials_file, tmp)
    brain.set_training_context(TrainingMaterials(tmp).context())
    brain.set_body_enabled(body_enabled)
    return brain


def load_turns(args) -> list[str]:
    lines: list[str] = []
    if args.file:
        for raw in Path(args.file).read_text(encoding="utf-8-sig").splitlines():
            text = raw.strip()
            if text and not text.startswith("#"):
                lines.append(text)
    lines += [t.strip() for t in (args.say or []) if t.strip()]
    return lines


STEP_LABEL = {"explain": "研修: 説明", "demonstrate": "研修: 見本", "ask": "研修: 質問"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?", help="人間側の発話を1行ずつ書いたファイル")
    p.add_argument("--say", action="append", help="発話を直接指定（複数可。- で始まる文はファイルで渡す）")
    p.add_argument("--connection", choices=("robot", "local"), default="robot",
                   help="robot=実機R1接続時の文脈（既定）、local=R1未接続のローカル試験パネルと同じ文脈")
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--reasoning-effort", default="minimal")
    p.add_argument("--materials-file", default=str(REPO / "logs" / "training" / "live_materials.json"))
    p.add_argument("--fsm", type=int, default=811, help="実機で読めていたモード番号（既定 811）")
    p.add_argument("--no-body", action="store_true", help="歩行無効の設定で試す（既定は 8770 と同じ有効）")
    p.add_argument("--out-dir", default=str(REPO / "logs" / "chat_verify"))
    p.add_argument("--label", default="", help="結果ファイル名に付ける短い名前")
    args = p.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    turns = load_turns(args)
    if not turns:
        raise SystemExit("発話がありません")
    body_enabled = not args.no_body
    brain = build_brain(args.model, args.reasoning_effort, Path(args.materials_file), body_enabled)
    session = Session(brain, args.connection, args.fsm, body_enabled)

    results = []
    conversation = 1
    for text in turns:
        if text == "---":
            session.reset()
            conversation += 1
            continue
        for row in session.handle(text):
            row["conversation"] = conversation
            results.append(row)
            if row["user"]:
                print(f"\n[{conversation}] 人間: {row['user']}")
            step = f"［{STEP_LABEL[row['training_step']]}］" if row["training_step"] else ""
            print(f"    R1{step} ({row['seconds']:.1f}s, {row['language']}): {row['say']}")
            print(f"    身ぶり: {row['gesture']} → {row['gesture_result']}"
                  + ("  ※発話を事実確認で置換" if row["replaced"] else "")
                  + (f"  ※研修段階で不許可の {row['training_blocked']} を止めた" if row["training_blocked"] else "")
                  + (f"  ※エラー {row['error']}" if row["error"] else ""))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"chat_verify_{stamp}" + (f"_{args.label}" if args.label else "")
    meta = {"model": args.model, "reasoning_effort": args.reasoning_effort, "connection": args.connection,
            "materials_file": str(Path(args.materials_file).name), "fsm": args.fsm,
            "body_enabled": body_enabled}
    (out / f"{name}.json").write_text(json.dumps({"meta": meta, "turns": results}, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    md = [f"# 会話検証 {stamp}", "",
          f"- モデル: {args.model}（reasoning {args.reasoning_effort}）／接続: "
          f"{'実機R1' if args.connection == 'robot' else 'R1未接続（ローカル）'}"
          f"／教材: {meta['materials_file']}／歩行: {'有効' if body_enabled else '無効'}", ""]
    current = 0
    for row in results:
        if row["conversation"] != current:
            current = row["conversation"]
            md += [f"## 会話 {current}", ""]
        if row["user"]:
            md += [f"**人間**: {row['user']}", ""]
        step = f"［{STEP_LABEL[row['training_step']]}］" if row["training_step"] else ""
        md += [f"**R1**{step}（{row['seconds']}秒・{row['language']}）: {row['say']}", "",
               f"身ぶり: `{row['gesture']}` → {row['gesture_result']}"
               + (f"（事実確認で置換。元の文: {row['replaced']['model_say']} / {row['replaced']['model_gesture']}）"
                  if row["replaced"] else "")
               + (f"（研修段階で不許可の {row['training_blocked']} を止めた）" if row["training_blocked"] else "")
               + (f"（エラー {row['error']}）" if row["error"] else ""), ""]
    (out / f"{name}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n保存: {out / (name + '.md')}")


if __name__ == "__main__":
    main()
