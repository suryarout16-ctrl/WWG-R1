"""GPT APIでも研修担当R1の文面・教材・動作検査を共用する。

公式Chat Completions仕様:
https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create
SDKは明示的な生成時までimportしない。モデル出力の解析と自己紹介・発話補正は
personas.LocalBrainに任せ、Strata用の推論オプションをGPT APIへ送らない。
"""
from __future__ import annotations

from personas import LocalBrain, MENTOR_PROMPT


class MentorAPIError(RuntimeError):
    """資格情報を含み得るSDKの例外本文は画面やログへ渡さない。"""
    def __init__(self, status=None):
        self.status = status if isinstance(status, int) else None
        super().__init__("GPT APIへの問い合わせに失敗しました" +
                         (f" (HTTP {self.status})" if self.status else ""))


class OpenAIMentorBrain(LocalBrain):
    BACKEND_LABEL = "OpenAIのGPT API（担当者が手動で切り替えた予備の応答先）"
    def __init__(self, client, model="gpt-5-mini", effort="minimal", timeout=20.0):
        super().__init__("", MENTOR_PROMPT, motion=True, model=model, timeout=timeout)
        self.client, self.effort = client, effort

    def available(self, timeout=2.0):
        try:
            self.client.models.retrieve(self.model, timeout=timeout)
            return True, f"GPT API {self.model}"
        except Exception as error:
            safe = MentorAPIError(getattr(error, "status_code", None))
            return False, str(safe)

    def _stream(self, body):
        # max_completion_tokensには推論分も含まれる。日本語の短い返答とJSON用に余裕を取る。
        kwargs = {"model": self.model, "messages": body["messages"], "stream": True,
                  "max_completion_tokens": 1200, "store": False, "timeout": self.timeout}
        if "response_format" in body:
            kwargs["response_format"] = body["response_format"]
        if self.effort and self.model.startswith(("gpt-5", "o1", "o3", "o4")):
            kwargs["reasoning_effort"] = self.effort
        stream = None
        try:
            stream = self.client.chat.completions.create(**kwargs)
            parts = []
            for chunk in stream:
                for choice in chunk.choices:
                    text = getattr(choice.delta, "content", None)
                    if text:
                        parts.append(text)
            return "".join(parts)
        except Exception as error:
            raise MentorAPIError(getattr(error, "status_code", None)) from None
        finally:
            if stream is not None:
                close = getattr(stream, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        # HTTP接続の後始末で資格情報を含む例外本文を露出させない。
                        pass
