"""R1 の人格と、つなぐ LLM。

  r1        : 標準の R1。OpenAI API（Responses）
  omoshiro  : おもしろR1。Z840 の Strata（OpenAI 互換 /v1/chat/completions、ローカル LLM）

人格の文面はここで編集する。Erika の人格・データ・声は使わない（別人格）。
Strata は Z840 の中（127.0.0.1:8081、Erika 用の常駐ゲートウェイ）でしか開いておらず、SSH のポート転送も
禁止されているので、SSH のコマンド実行で Z840 上の中継を呼ぶ（ssh_relay.py）。Z840 の設定は変えない。
  環境変数 R1_Z840_HOST / R1_Z840_USER / R1_Z840_KEY_DIR / R1_Z840_HOSTKEY_ALIAS（web_ui の --z840-* でも可）
HTTP で直接届く OpenAI 互換サーバーなら R1_STRATA_URL / --strata-url を使う。
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

import r1_chat as rc
from ssh_relay import RelayError, SshRelay

OMOSHIRO_PROMPT = """あなたはヒューマノイドロボット「R1」の中にいる、おしゃべり好きでちょっとおちゃめな人格「おもしろR1」です。
ハッカソン会場で来場者と話しています。
- 一人称は「僕」。明るく、軽いダジャレやツッコミを時々まぜる。相手をからかったり、傷つけたりはしない
- 返答は日本語で1〜2文、60文字程度まで。音声で読み上げるので短く
- 記号、絵文字、箇条書き、URL、英字の略語は使わない。数字は読みやすく
- 知らないことは知らないと言う。作り話で答えない
- 考えた過程は書かない。返事だけを書く"""

PERSONAS = {
    "r1": {"label": "R1（標準・OpenAI）", "backend": "openai", "prompt": rc.SYSTEM_PROMPT},
    "omoshiro": {"label": "おもしろR1（Z840 Strata）", "backend": "strata", "prompt": OMOSHIRO_PROMPT},
}

THINK = re.compile(r"<think>.*?</think>", re.S)


def http_json(url: str, body: dict | None = None, timeout: float = 60.0, api_key: str = "") -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"}
    if api_key:   # Strata をキー付きで公開している場合（環境変数 R1_STRATA_KEY）
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class LocalBrain:
    """OpenAI 互換の /v1/chat/completions（Strata の serve/server.py、llama.cpp server など）で話す。

    会話履歴はこちらで持つ（直近 MAX_TURNS 往復）。身ぶりは JSON（response_format json_schema）で受け取り、
    サーバーが JSON 指定を受け付けなければ文だけで話す。
    """
    MAX_TURNS = 8

    def __init__(self, base_url: str, prompt: str, motion: bool = True, model: str = "", timeout: float = 60.0,
                 api_key: str = "", relay: SshRelay | None = None):
        self.base_url, self.motion, self.model, self.timeout = base_url.rstrip("/"), motion, model, timeout
        self.api_key = api_key or os.environ.get("R1_STRATA_KEY", "")
        self.instructions = prompt + (rc.MOTION_RULE if motion else rc.NO_MOTION_RULE)
        self.history: list[dict] = []
        self.json_ok = True     # JSON 指定が通らなかったら False にして以降は文だけ
        self.relay = relay      # あれば SSH 経由（Z840）、無ければ base_url に HTTP

    def reset(self) -> None:
        self.history = []

    def available(self, timeout: float = 2.0) -> tuple[bool, str]:
        try:
            if self.relay:
                data = self.relay.request({"op": "models"}, max(timeout, 8.0))["data"]
            else:
                data = http_json(f"{self.base_url}/models", timeout=timeout, api_key=self.api_key)
            models = [m.get("id") for m in data.get("data", [])]
            if models and not self.model:
                self.model = models[0]
            return True, ", ".join(m for m in models if m) or "（モデル名なし）"
        except Exception as error:
            return False, f"{type(error).__name__}: {str(error)[:120]}"

    def _stream(self, body: dict) -> str:
        """SSE（data: {...}）を読み、本文（content）だけをつなげて返す。思考（reasoning_content）は捨てる。"""
        if self.relay:
            return self.relay.request({"op": "chat", "body": body}, self.timeout)["content"]
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", json.dumps(body).encode("utf-8"), headers)
        content = ""
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    raise RuntimeError(f"Strata のエラー: {str(chunk['error'])[:200]}")
                for choice in chunk.get("choices", []):
                    content += (choice.get("delta") or {}).get("content") or ""
        return content

    def reply(self, user_text: str) -> tuple[str, str]:
        messages =[{"role": "system", "content": self.instructions}, *self.history,
                    {"role": "user", "content": user_text}]
        if not self.model:
            self.available()
        # Z840 の Erika 用ゲートウェイは stream: true とモデル名の一致が必須。思考は止める（1.8 秒前後で返る）
        body = {"model": self.model or "local", "messages": messages, "max_tokens": 300, "temperature": 0.7,
                "stream": True, "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
        if self.motion and self.json_ok:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "r1_reply", "strict": True,
                "schema": {"type": "object", "additionalProperties": False, "required": ["say", "gesture"],
                           "properties": {"say": {"type": "string"},
                                          "gesture": {"type": "string", "enum": ["none", *rc.GESTURES]}}}}}
        t0 = time.time()
        try:
            content = self._stream(body)
        except (urllib.error.HTTPError, RelayError) as error:
            code = getattr(error, "code", None) or getattr(error, "status", None)
            if "response_format" in body and "structured_output" in str(error):
                # モデルが JSON を作れなかった回だけ、身ぶりなしの文で答え直す
                rc.log("  [LLM] JSON にできなかったので、この回は文だけで答え直します")
                body.pop("response_format")
                content = self._stream(body)
            elif "response_format" in body and code in (400, 422, 501):
                rc.log(f"[LLM] Strata は JSON 指定を受け付けないので、文だけで話します ({code}: {error})")
                self.json_ok = False
                return self.reply(user_text)
            else:
                raise
        out = THINK.sub("", content).strip()
        say, gesture = out, "none"
        if "response_format" in body:
            try:
                parsed = json.loads(out[out.find("{"):out.rfind("}") + 1])
                say, gesture = str(parsed.get("say", "")).strip(), str(parsed.get("gesture", "none"))
            except (ValueError, AttributeError):
                pass
        if not say:
            raise RuntimeError(f"返答が空でした ({time.time() - t0:.1f}s)")
        # JSON で答えさせているときは、履歴の返事も JSON の形で残す（文だけの履歴を真似て JSON を崩すのを防ぐ）
        said = json.dumps({"say": say, "gesture": gesture}, ensure_ascii=False) if self.motion and self.json_ok else say
        self.history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": said}]
        self.history = self.history[-self.MAX_TURNS * 2:]
        return say, gesture if gesture in rc.GESTURES else "none"
