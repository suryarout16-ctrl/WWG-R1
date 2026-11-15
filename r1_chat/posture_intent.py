"""起立の直接依頼だけを認識する。状態の推測やロボットへの送信はしない。"""
from __future__ import annotations

import re
import unicodedata


_JA_ACTION = r"(?:起立して(?:ください)?|立ち上がって(?:ください)?)"
_EN_ACTION = r"(?:please )?stand up(?: please)?"
_JA_UNKNOWN = re.compile(_JA_ACTION)
_JA_SUPINE = re.compile(r"仰向け(?:から|で)\s*(?:,\s*)?" + _JA_ACTION)
_JA_READY = re.compile(r"仰向けで\s*準備確認済み\s*,\s*" + _JA_ACTION)
_JA_FULL_READY = re.compile(
    r"仰向けに寝ています\s*\.\s*起立準備確認済み\s*\.\s*" + _JA_ACTION
)
_EN_UNKNOWN = re.compile(_EN_ACTION, re.I)
_EN_SUPINE = re.compile(r"supine\s*,\s*" + _EN_ACTION, re.I)
_EN_READY = re.compile(r"supine\s*,\s*preparation confirmed\s*,\s*" + _EN_ACTION, re.I)
_LINE_BREAK = re.compile(r"[\r\n\v\f\u0085\u2028\u2029]")
_QUOTES = frozenset('"\'`「」『』“”‘’«»')


def parse_stand_request(text: str) -> dict | None:
    """許可済みの短い依頼の形だけを返す。引用・質問・条件付き依頼は扱わない。

    posture=Noneやready=Falseも、実行せず不足条件を説明するための認識結果。
    ready=Trueは入力文の明示申告であり、姿勢の実測や安全確認を代行しない。
    """
    if not isinstance(text, str) or not text or len(text) > 120 or _LINE_BREAK.search(text):
        return None
    value = unicodedata.normalize("NFKC", text)
    if "?" in value or any(mark in value for mark in _QUOTES):
        return None
    value = value.translate(str.maketrans({"、": ",", "。": "."}))
    value = re.sub(r"\s+", " ", value).strip()
    # 一つの終止符・感嘆符だけを許可し、複数命令の区切りは残す。
    value = re.sub(r"[.!]$", "", value).strip()
    if _JA_READY.fullmatch(value) or _JA_FULL_READY.fullmatch(value) or _EN_READY.fullmatch(value):
        return {"posture": "supine", "ready": True}
    if _JA_SUPINE.fullmatch(value) or _EN_SUPINE.fullmatch(value):
        return {"posture": "supine", "ready": False}
    if _JA_UNKNOWN.fullmatch(value) or _EN_UNKNOWN.fullmatch(value):
        return {"posture": None, "ready": False}
    return None
