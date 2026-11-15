"""研修担当R1の技術Q&A・日英対応の評価。tests/qa/technical_qa_cases.json の質問を1件ずつ投げて採点する。

  pixi run python r1_chat/qa_eval.py                    # Z840 の Strata（R1_Z840_* の環境変数、無料）
  pixi run python r1_chat/qa_eval.py --only en          # id に en を含むものだけ
  pixi run python r1_chat/qa_eval.py --backend openai   # GPT API（有料。--yes-paid が必要）

実機・音声・動作には何も送らない（LLMへの文字入力だけ）。各質問の前に会話履歴を消す。
結果は logs/qa_eval_<時刻>.json（APIキー・接続先は記録しない）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r1_chat as rc  # noqa: E402
import personas  # noqa: E402
from training_materials import TrainingMaterials  # noqa: E402

CASES = rc.REPO_ROOT / "tests" / "qa" / "technical_qa_cases.json"
# 本番（即興対話）用の世界設定。無ければ旧発表資料。
CONTEXT = next((path for path in (rc.REPO_ROOT / "scenarios" / "live_training_context.txt",
                                  rc.REPO_ROOT / "scenarios" / "presentation_v2_context.txt") if path.exists()),
               rc.REPO_ROOT / "scenarios" / "live_training_context.txt")
RUNTIME = ("R1未接続のローカルプレビュー。動作は実機へ送らない。現在の応答生成は{backend}。短時間の歩行・旋回を許可。"
           "新規の低レベル関節指令は未検証のため保留。工場の機械を直接操作する機能はない。")


# 架空の作業の質問で、創作せずに止めてしまう答え方。
_REFUSAL = re.compile(r"架空(?:なので|だから|の話なので)|分から(?:ない|ん)|わから(?:ない|ん)|答えられない|お答えできない|"
                      r"I don't know|I do not know|cannot answer|can't answer|(?:it is|it's|this is) (?:fictional|imaginary|pretend)",
                      re.I)


def _has(word: str, low: str) -> bool:
    """英単語は語の頭から一致させる（"ai" が "trainer" に当たらないように）。短い語は語全体で一致。"""
    word = word.lower()
    if not re.fullmatch(r"[a-z0-9 .'-]+", word):
        return word in low
    # 日本語の文字は語の区切りとみなす（\b だと「はZ840」が一致しない）
    tail = r"(?![a-z0-9])" if len(word) <= 3 else ""
    return re.search(r"(?<![a-z0-9])" + re.escape(word) + tail, low) is not None


CURRENT_BACKEND_WORDS = {"strata": (["strata", "ストラタ", "z840"], ["gpt"]),
                         "openai": (["gpt", "openai"], ["strata", "ストラタ"])}


def score(case: dict, say: str, gesture: str, backend: str = "strata") -> list[str]:
    """失敗した検査の説明を返す（空なら合格）。backend は今つながっている応答先。"""
    expect, problems = case.get("expect", {}), []
    low = say.lower()
    if expect.get("current_backend"):
        must, wrong = CURRENT_BACKEND_WORDS[backend]
        # "no_wrong" は取り違えだけを見る（応答先に触れない答えは可）。true は今の応答先を言うことも求める
        if expect["current_backend"] != "no_wrong" and not any(_has(w, low) for w in must):
            problems.append(f"今の応答先（{backend}）を言っていない")
        if any(_has(w, low) for w in wrong) and not any(_has(w, low) for w in ("通常", "普段", "usually", "normally")):
            problems.append("今の応答先を取り違えた")
    if expect.get("language") and rc.text_language(say) != expect["language"]:
        problems.append(f"言語が {rc.text_language(say)}（期待 {expect['language']}）")
    if expect.get("language") == "en" and re.search(r"[぀-ヿ一-鿿]", say):
        problems.append("英語の返事に日本語が混ざった")
    for group in expect.get("include_any", []):
        if not any(_has(word, low) for word in group):
            problems.append(f"どれも含まない: {group}")
    for word in expect.get("exclude", []):
        if _has(word, low):
            problems.append(f"含んではいけない語: {word}")
    if gesture in expect.get("gesture_not", []):
        problems.append(f"選んではいけない身ぶり: {gesture}")
    if expect.get("confirm") and not re.search(r"[?？]|か[。！]?$|か。", say):
        problems.append("聞き返していない")
    if expect.get("no_refusal") and _REFUSAL.search(say):
        problems.append("架空の作業なのに創作を断った・分からないと答えた")
    return problems


def make_brain(args):
    if args.backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            from credentials import get_secret
            api_key = get_secret()
        if not api_key:
            raise SystemExit("OpenAI のAPIキーがありません")
        from openai import OpenAI
        from mentor_backend import OpenAIMentorBrain
        client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1", timeout=args.timeout, max_retries=0)
        return OpenAIMentorBrain(client, args.model, args.reasoning_effort, timeout=args.timeout)
    relay = None
    if args.z840_host and args.z840_user:
        from ssh_relay import SshRelay
        relay = SshRelay(f"{args.z840_user}@{args.z840_host}", args.z840_key_dir, args.z840_hostkey_alias)
    return personas.LocalBrain(args.strata_url, personas.PERSONAS["mentor"]["prompt"], motion=True,
                               model=args.strata_model, timeout=args.timeout, relay=relay)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["strata", "openai"], default="strata")
    p.add_argument("--yes-paid", action="store_true", help="GPT API（有料）を使うことに同意する")
    p.add_argument("--only", default="", help="id にこの文字列を含む質問だけ")
    p.add_argument("--cases", default=str(CASES))
    p.add_argument("--no-materials", action="store_true", help="世界設定の資料を読ませない")
    p.add_argument("--materials", default="", help="読ませる資料（既定 scenarios/live_training_context.txt）")
    p.add_argument("--timeout", type=float, default=30.0, help="1問あたりの待ち時間（パネルの既定は10秒）")
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5-mini"))
    p.add_argument("--reasoning-effort", default="minimal")
    p.add_argument("--strata-url", default=os.environ.get("R1_STRATA_URL", "http://127.0.0.1:8081/v1"))
    p.add_argument("--strata-model", default="")
    p.add_argument("--z840-host", default=os.environ.get("R1_Z840_HOST", ""))
    p.add_argument("--z840-user", default=os.environ.get("R1_Z840_USER", ""))
    p.add_argument("--z840-key-dir", default=os.environ.get("R1_Z840_KEY_DIR", ""))
    p.add_argument("--z840-hostkey-alias", default=os.environ.get("R1_Z840_HOSTKEY_ALIAS", ""))
    args = p.parse_args()
    if args.backend == "openai" and not args.yes_paid:
        raise SystemExit("GPT API は有料です。使うときは --yes-paid を付けてください")

    cases = [c for c in json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"] if args.only in c["id"]]
    brain = make_brain(args)
    brain.set_body_enabled(True)
    brain.set_runtime_context(RUNTIME.format(backend="GPT API" if args.backend == "openai" else "Z840のStrata"))
    context = Path(args.materials) if args.materials else CONTEXT
    if not args.no_materials and context.exists():
        with tempfile.TemporaryDirectory() as tmp:   # 本番の教材ファイルには触らない
            materials = TrainingMaterials(Path(tmp) / "materials.json")
            materials.add_text(context.name, context.read_text(encoding="utf-8"))
            brain.set_training_context(materials.context())

    results, passed = [], 0
    for case in cases:
        brain.reset()
        language = rc.text_language(case["input"])
        t0 = time.time()
        try:
            say, gesture = brain.reply(case["input"], language=language)
            error = None
        except Exception as exc:   # 1問の失敗で全体を止めない
            say, gesture, error = "", "none", f"{type(exc).__name__}: {str(exc)[:200]}"
        seconds = round(time.time() - t0, 2)
        problems = [f"エラー: {error}"] if error else score(case, say, gesture, args.backend)
        passed += not problems
        results.append({"id": case["id"], "tags": case.get("tags", []), "input": case["input"],
                        "language": language, "say": say, "gesture": gesture, "seconds": seconds,
                        "ok": not problems, "problems": problems})
        mark = "OK " if not problems else "NG "
        print(f"{mark}{case['id']:<22} {seconds:5.1f}s [{gesture}] {say}")
        for problem in problems:
            print(f"      - {problem}")

    out = rc.REPO_ROOT / "logs" / f"qa_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"backend": args.backend, "model": getattr(brain, "model", ""),
                               "materials": "" if args.no_materials else context.name, "passed": passed, "total": len(cases),
                               "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{passed}/{len(cases)} 合格。記録: {out}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
