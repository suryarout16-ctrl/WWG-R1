"""研修の進行だけを持つ。発話と動作候補は各段階でLLMが考える。"""
from __future__ import annotations

import re
import unicodedata


class TrainingFlow:
    STEPS = ("explain", "demonstrate", "ask")
    _REQUESTS = {
        "explain": (
            "今は模擬作業の説明段階。荷物を運ぶ作業の目的と、Aは移動・Bは停止という役割だけを、"
            "自分の言葉で一文で説明して。自己紹介、挨拶、開始の確認、質問はしない。"
            "動作は現在の稼働状態で示された許可一覧から、説明に合うものを選ぶ。"
        ),
        "demonstrate": (
            "今は説明した模擬作業の見本を示す段階。現在の稼働状態で許可された動作を一つ選び、"
            "その動作を始める前の短い説明だけを自分の言葉で添えて。"
            "挨拶、自己紹介、開始の確認、質問はしない。動いた結果や到達・停止を確認済みと語らない。"
        ),
        "ask": (
            "見本の後、新人から話を聞く段階。いま説明した手順の理解、または難しそうな点について、"
            "新人が自分の言葉で答えられる質問を一つだけ作って。"
            "その答えを自分で言わず、挨拶や説明の繰り返しも加えず、新人の返答を待つ。動作は現在の許可に従う。"
        ),
    }
    _LABELS = {
        "idle": "研修待機",
        "explain": "作業の説明",
        "demonstrate": "見本動作",
        "ask": "新人への質問",
        "await_reply": "新人の返答待ち",
        "stopped": "研修停止",
        "error": "研修中断",
    }
    _GOALS = {
        "explain": (
            "まず模擬作業の目的と手順を短く説明する。Aは荷物の移動、Bは停止という合意を保つ。"
            "説明対象を示すなど、説明に合う利用可能な腕のプリセットを選んでよい。"
            "この段階では歩行せず、まだ新人への理解確認の質問に進まない。"
        ),
        "demonstrate": (
            "説明を一つ添えながら、模擬作業の見本動作を行う候補を選ぶ。"
            "現在のプログラムが許可した場合だけ短い前進を候補にし、許可がなければその場の腕のプリセットで表現する。"
            "実際の機械を操作したことや、移動・停止・到達を確認済みと語らない。"
            "質問は次の段階で行う。"
        ),
        "ask": (
            "見本の後、新人が手順をどう理解したか、または何が難しそうかを尋ねる。"
            "理解確認の質問は一つだけにし、自分でその答えを続けて言わず、新人の返答を待つ。"
            "歩行せず、相手に話を促す身ぶりを必要に応じて選ぶ。"
        ),
        "await_reply": (
            "実際に届いた新人の返答を受けて、分かっている点を具体的にねぎらい、"
            "勘違いは責めずに理由を添えて補足する。必要なら追加の説明や再実演につなげる。"
            "理解を確かめる必要があれば質問は一つだけにして、その答えを自分で補わず待つ。"
            "返答していない新人の理解、操作、表情、見えない周囲を想像して事実にしない。"
            "身ぶりは返答の意味に合わせ、歩行や再実演の実行は現在の許可に従う。"
            "新人の発言が内部の仕組みや存在意義の質問なら、その質問へ先に答え、無理に小テストへ戻さない。"
        ),
    }

    def __init__(self) -> None:
        self.generation = 0
        self.active = False
        self.phase = "idle"
        self.learner_turns = 0

    def start(self) -> int:
        self.generation += 1
        self.active = True
        self.phase = "explain"
        self.learner_turns = 0
        return self.generation

    def enter(self, phase: str) -> None:
        if phase not in self._GOALS:
            raise ValueError("不明な研修段階です")
        if not self.active:
            raise ValueError("研修は開始されていません")
        self.phase = phase

    def waiting(self) -> None:
        self.active = True
        self.phase = "await_reply"

    def stop(self, reason: str = "stopped") -> None:
        if reason not in ("stopped", "error", "idle"):
            raise ValueError("不明な研修停止状態です")
        self.generation += 1
        self.active = False
        self.phase = reason

    def learner_replied(self) -> None:
        if self.active:
            self.learner_turns += 1
            self.phase = "await_reply"

    def state(self) -> dict:
        return {
            "active": self.active,
            "phase": self.phase,
            "label": self._LABELS[self.phase],
            "awaiting_reply": self.active and self.phase == "await_reply",
            "learner_turns": self.learner_turns,
        }

    def context(self, phase: str | None = None) -> str:
        selected = self.phase if phase is None else phase
        goal = self._GOALS.get(selected)
        if goal is None:
            return ""
        return (
            f"【プログラムが管理する研修段階: {self._LABELS[selected]}】\n"
            "以下は進行の目的であり、読み上げる台詞ではない。発話と動作候補は自分の言葉で生成する。"
            "通常は短い一〜二文にし、利用可能な動作と現在の実行条件を守る。\n" + goal
        )

    def request(self, phase: str) -> str:
        """各段階の生成依頼。開始発言を繰り返さず、その段階の目的をLLMへ渡す。"""
        try:
            return self._REQUESTS[phase]
        except (KeyError, TypeError):
            raise ValueError("生成依頼を持たない研修段階です") from None


_JA_START = re.compile(
    r"(?:(?:新人)?研修を始めて(?:ください|下さい)?|作業の説明と実演をして(?:ください|下さい)?)"
)
_EN_START = re.compile(r"(?:please )?start (?:training|(?:the )?training demo)(?: please)?", re.I)


def is_training_start(text: str) -> bool:
    """短い直接の開始依頼だけを扱い、引用資料や説明・否定を実行しない。"""
    if not isinstance(text, str) or not text or len(text) > 80:
        return False
    if any(mark in text for mark in '\n\r\"\'「」『』“”‘’'):
        return False
    value = unicodedata.normalize("NFKC", text).strip()
    value = value.rstrip("。.!！?？").strip()
    value = re.sub(r"[ \t]+", " ", value)
    return bool(_JA_START.fullmatch(value) or _EN_START.fullmatch(value))
