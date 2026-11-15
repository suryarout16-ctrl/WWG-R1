"""日本語・英語の切り替え。研修担当R1の人格・事実・安全条件は言語に関係なく同じ。

- 返答言語は、音声なら音声認識が判定した言語、文字なら文字の種類（r1_chat.text_language）で決める
- 英語の相手はなまりがあることを前提に、短く平易な英語で話し、機械操作の曖昧さは推測せず聞き返す
- 歩行の直接依頼（英語）は body_control.requested_body_actions と同じく、短い命令文だけを許可する
"""
from __future__ import annotations

import re
import unicodedata

ENGLISH_REPLY_RULE = """
【今回の返答言語: 英語 / Reply language for this turn: English】
- The learner is speaking English. They may have an Indian or other accent, and the speech recognition text may contain mistakes.
- Write the "say" text in English only. Do not use Japanese in "say". The Japanese rules above about "日本語で" and "英字の略語" do not apply to this turn.
- Keep the same character: a friendly, experienced factory trainer who teaches new workers. Use "I". Warm and calm. Never mock anyone's accent or English.
- Use simple, clear English for non-native listeners: 1 to 2 short sentences (under 15 words each, about 35 words total), common words, no idioms or slang. Button names A, B and C are fine.
- Exception: for questions about how R1 itself works, use up to 3 short sentences about the parts relevant to the question. Name the current AI and model only from the measured current-backend information above; the usual configuration is not necessarily active. Keep local preview, virtual R1 and physical R1 distinct.
- For self-introductions and purpose questions, use up to 4 short sentences: teach factory newcomers, prevent accidents from operating mistakes, and explain the roles separately. The AI chooses words and motion candidates; the program turns them into commands; permission and robot-state checks decide whether commands may run. Do not claim automatic detection of surrounding hazards. Do not add unrelated implementation details. For questions about walking limits, mention the 2-second limit and final zero-speed command.
- When explaining those roles, use "the AI" as the subject that chooses words and gestures. R1 is the robot expressing its output, not a separate controller deciding what the AI says. Never reverse that relationship by saying "I decide what the AI says".
- If a request about a machine operation, a button (A, B, C), start or stop is unclear, or a word looks misheard, do not guess. Ask one short confirmation question, for example "Did you mean button A?"
- All facts, safety limits, gesture rules and the reference documents above still apply. Translate them; do not change them.
- The three conversation areas above apply in English too: for the pretend factory work you may invent details, examples and light jokes (keep A = move, B = stop and other agreed settings); for questions about R1's real system use only the given facts and say you are not sure when a fact is missing; explain your purpose in your own words."""

CONFIRM_RULE = """
- 機械の操作・ボタン（A、B、C）・開始や停止についての指示が曖昧なとき、聞き間違いらしい語があるときは、推測で進めず短く1つだけ聞き返す"""

LOW_CONFIDENCE_RULE = {
    "ja": "\n【音声認識の注意】今回の発言は音声認識の確信度が低く、聞き間違いを含むかもしれない。機械の操作・ボタン・開始停止に関わる内容なら、推測せず短く聞き返す。",
    "en": "\n[Speech recognition note] This turn was recognized with low confidence and may contain misheard words. If it is about a machine operation, a button, start or stop, do not guess: ask one short confirmation question in English.",
}

PLAIN_REPLY_RULE_EN = """
This time, output only the English text to be spoken, without a gesture. Usually 1 to 2 short sentences.
Do not output JSON, the field names say/gesture, code blocks or your reasoning."""

BODY_ANNOUNCEMENTS_EN = {
    "move_forward": "A short step forward. Let me check the robot status first.",
    "move_backward": "A short step back. Let me check the robot status first.",
    "move_left": "A short step to the left. Let me check the robot status first.",
    "move_right": "A short step to the right. Let me check the robot status first.",
    "turn_left": "A short turn to the left. Let me check the robot status first.",
    "turn_right": "A short turn to the right. Let me check the robot status first.",
    "stop_move": "You asked me to stop. Please check with your eyes that I have stopped.",
}

ERROR_REPLY = {
    "ja": "すまん、通信がうまくいかないな。少し待って、もう一度聞かせてくれ。",
    "en": "Sorry, I have a connection problem. Please wait a moment and say it again.",
}
NOT_HEARD = {
    "ja": "聞き取れませんでした。もう一度どうぞ",
    "en": "I could not hear that. Please say it again.",
}

# 英語の直接の移動依頼。日本語版（r1_motion/body_control.requested_body_actions）と同じく、
# 質問・条件・否定・説明・引用の文では動かさず、短い命令文だけを許可する。
_EN_SKIP = re.compile(
    r"\b(?:how|what|why|when|if|can you|could you|would you|will you|should|explain|tell me|mean|means|"
    r"say|said|says|read|translate|later|after|before|minutes?|seconds?|pressed|push(?:ed)?|button|"
    r"don't|do not|never|no need|not)\b|[\"“”「」『』]", re.I)
_EN_PREFIX = r"(?:(?:hey|ok|okay)[,\s]+)?(?:r ?1[,\s]+)?(?:please[,\s]+)?"
_EN_SUFFIX = r"(?:[,\s]+(?:please|now|a little|a bit|slowly|one step))*"
_EN_ACTIONS = (
    ("stop_move", r"(?:stop|stop walking|stop moving|halt|freeze)"),
    ("move_forward", r"(?:(?:walk|move|go|step)\s+(?:a little\s+|a bit\s+|slowly\s+)?(?:forward|ahead|to the front)|come forward)"),
    ("move_backward", r"(?:(?:walk|move|go|step)\s+(?:a little\s+|a bit\s+|slowly\s+)?back(?:ward|wards)?|back up)"),
    ("move_left", r"(?:(?:walk|move|step)\s+(?:a little\s+|a bit\s+)?(?:to the\s+)?left)"),
    ("move_right", r"(?:(?:walk|move|step)\s+(?:a little\s+|a bit\s+)?(?:to the\s+)?right)"),
    ("turn_left", r"(?:turn\s+(?:a little\s+|a bit\s+)?(?:to the\s+)?left)"),
    ("turn_right", r"(?:turn\s+(?:a little\s+|a bit\s+)?(?:to the\s+)?right)"),
)


def requested_body_actions_en(text: str) -> set[str]:
    """英語の短い直接依頼だけを許可する（例: "Please walk forward", "Stop!"）。"""
    if not isinstance(text, str) or not text.isascii():
        return set()
    allowed = set()
    for clause in re.split(r"[.!?\n;]", unicodedata.normalize("NFKC", text).lower()):
        clause = re.sub(r"\s+", " ", clause).strip(" ,")
        if not clause:
            continue
        stop_only = re.fullmatch(_EN_PREFIX + r"(?:stop|stop walking|stop moving|halt|freeze)" + _EN_SUFFIX, clause)
        if stop_only:
            allowed.add("stop_move")
            continue
        if _EN_SKIP.search(clause):
            continue
        for action, pattern in _EN_ACTIONS:
            if re.fullmatch(_EN_PREFIX + pattern + _EN_SUFFIX, clause):
                allowed.add(action)
    return allowed
