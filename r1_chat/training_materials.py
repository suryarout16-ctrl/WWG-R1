"""研修教材をPC内だけに保存する。対応形式はUTF-8の.txt/.mdと貼り付け文章。"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


class TrainingMaterials:
    MAX_DOCUMENTS = 8
    MAX_DOCUMENT_CHARS = 8000
    MAX_TOTAL_CHARS = 12000
    SUPPORTED_EXTENSIONS = {".txt", ".md"}

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else Path(__file__).resolve().parent.parent / "logs" / "training" / "materials.json"
        self._documents: list[dict] = []
        if self.path.exists():
            self._load()

    @staticmethod
    def _name(name: str) -> str:
        if not isinstance(name, str):
            raise ValueError("教材名を入力してください")
        name = name.replace("\\", "/").rsplit("/", 1)[-1].strip()
        if not name or len(name) > 120 or any(ord(c) < 32 for c in name):
            raise ValueError("教材名は120文字以内で入力してください")
        return name

    @classmethod
    def _text(cls, text: str) -> str:
        if not isinstance(text, str):
            raise ValueError("教材は文章で入力してください")
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text or "\x00" in text:
            raise ValueError("教材に読み取れる文章がありません")
        if len(text) > cls.MAX_DOCUMENT_CHARS:
            raise ValueError(f"教材1件は{cls.MAX_DOCUMENT_CHARS:,}文字以内にしてください")
        return text

    @classmethod
    def _validate_collection(cls, docs: list[dict]) -> None:
        if len(docs) > cls.MAX_DOCUMENTS:
            raise ValueError(f"教材は{cls.MAX_DOCUMENTS}件までです。不要な教材を削除してください")
        if sum(len(d["text"]) for d in docs) > cls.MAX_TOTAL_CHARS:
            raise ValueError(f"教材の合計は{cls.MAX_TOTAL_CHARS:,}文字以内にしてください")

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("version") != 1 or not isinstance(raw.get("documents"), list):
                raise ValueError("保存形式が違います")
            docs = []
            ids = set()
            for doc in raw["documents"]:
                if (not isinstance(doc, dict) or not isinstance(doc.get("id"), str) or not doc["id"]
                        or doc["id"] in ids or not isinstance(doc.get("added_at"), str)):
                    raise ValueError("教材情報が不正です")
                docs.append({"id": doc["id"], "name": self._name(doc["name"]),
                             "text": self._text(doc["text"]), "added_at": doc["added_at"]})
                ids.add(doc["id"])
            self._validate_collection(docs)
            self._documents = docs
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise ValueError("保存済み教材を読み込めません。教材ファイルを確認してください") from error

    def _save(self, docs: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            temporary.write_text(json.dumps({"version": 1, "documents": docs}, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()
        self._documents = docs

    @staticmethod
    def _metadata(doc: dict) -> dict:
        return {"id": doc["id"], "name": doc["name"], "chars": len(doc["text"]), "added_at": doc["added_at"]}

    def documents(self) -> list[dict]:
        return [self._metadata(doc) for doc in self._documents]

    def add_text(self, name: str, text: str) -> dict:
        doc = {"id": uuid.uuid4().hex, "name": self._name(name), "text": self._text(text),
               "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        docs = [*self._documents, doc]
        self._validate_collection(docs)
        self._save(docs)
        return self._metadata(doc)

    def add_file(self, name: str, data: bytes) -> dict:
        name = self._name(name)
        if Path(name).suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError("対応形式はUTF-8の.txtと.mdです。PDFやWordは文章を貼り付けてください")
        if not isinstance(data, bytes) or len(data) > self.MAX_DOCUMENT_CHARS * 4:
            raise ValueError("教材ファイルが大きすぎるか、読み取れません")
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError("UTF-8形式で保存した.txtまたは.mdを選んでください") from error
        return self.add_text(name, text)

    def remove(self, document_id: str) -> None:
        docs = [doc for doc in self._documents if doc["id"] != document_id]
        if len(docs) == len(self._documents):
            raise ValueError("指定された教材が見つかりません")
        self._save(docs)

    def clear(self) -> None:
        self._save([])

    def context(self) -> str:
        if not self._documents:
            return ""
        return json.dumps([{"name": doc["name"], "text": doc["text"]} for doc in self._documents], ensure_ascii=False)
