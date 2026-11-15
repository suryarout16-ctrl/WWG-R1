"""faster-whisper worker (small / ja / VAD, beam 5).

Runs as a subprocess of r1_chat.py (same pixi env by default).
Protocol: one JSON per line on stdin {"id", "audioPath"} -> one JSON per line on stdout
{"id", "type": "completed"|"error", "text", "confidence", "language", "language_prob", "avg_logprob"}.
Logs go to stderr.

--language は ja / en、または "auto"（= "ja,en"）。複数指定のときは Whisper の言語判定を
その言語だけに絞って比べる（全言語で判定すると、なまりのある英語がヒンディー語などに化けるため）。
先頭の言語を優先し、ほかの言語は正規化した確率が --switch-threshold 以上のときだけ選ぶ。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback

DEFAULT_SWITCH_THRESHOLD = 0.65


def parse_languages(value: str) -> list[str]:
    """'ja' / 'en' / 'auto' / 'ja,en' を言語コードの並びにする（先頭が優先）。"""
    value = (value or "ja").strip().lower()
    if value == "auto":
        return ["ja", "en"]
    langs = [v.strip() for v in value.split(",") if v.strip()]
    if not langs or any(not v.isalpha() or len(v) > 3 for v in langs):
        raise ValueError(f"--language が不正です: {value!r}")
    return list(dict.fromkeys(langs))


def choose_language(all_probs, allowed: list[str], threshold: float = DEFAULT_SWITCH_THRESHOLD) -> tuple[str, float]:
    """許可した言語だけで確率を正規化し、優先言語（先頭）以外はしきい値を超えたときだけ選ぶ。

    all_probs: [(言語, 確率), ...]（faster-whisper の detect_language の 3 つ目）
    戻り値: (言語, 許可した言語の中での正規化確率)
    """
    probs = {lang: 0.0 for lang in allowed}
    for lang, p in all_probs:
        if lang in probs:
            probs[lang] = max(probs[lang], float(p))
    total = sum(probs.values())
    if total <= 0:
        return allowed[0], 0.0
    norm = {lang: p / total for lang, p in probs.items()}
    primary = allowed[0]
    best = max((lang for lang in allowed[1:]), key=lambda lang: norm[lang], default=primary)
    if best != primary and norm[best] >= threshold:
        return best, norm[best]
    return primary, norm[primary]


def looks_like_katakana_english(text: str) -> bool:
    """日本語として文字にした結果がほぼカタカナだけ（例: 「プリーズイントロデュースユアセルフ」）か。

    強いなまりの英語を日本語と判定すると、助詞のないカタカナの羅列になる。日本語の発話は
    「Aボタンを押して」のように、ひらがな・漢字がほぼ必ず混ざる。
    """
    jp = re.findall(r"[぀-ヿ一-鿿]", text or "")
    kata = [c for c in jp if "゠" <= c <= "ヿ"]
    return len(kata) >= 8 and len(kata) >= 0.9 * len(jp)


def send(message: dict) -> None:
    print(json.dumps(message, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--language", default="ja", help="ja / en / auto（= ja,en）/ ja,en など")
    parser.add_argument("--switch-threshold", type=float, default=DEFAULT_SWITCH_THRESHOLD,
                        help="先頭以外の言語へ切り替える正規化確率（既定 0.65）")
    parser.add_argument("--vad-filter", action="store_true")
    parser.add_argument("--dll-dir", action="append", default=[],
                        help="cuBLAS 12 / cuDNN 9 の DLL があるフォルダ（Windows の GPU 用）")
    args = parser.parse_args()

    for d in args.dll_dir:
        if os.path.isdir(d):
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]

    from faster_whisper import WhisperModel, decode_audio

    languages = parse_languages(args.language)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    def run(path: str) -> dict:
        audio = decode_audio(path, sampling_rate=16000)
        language, language_prob = languages[0], 1.0
        if len(languages) > 1:
            _lang, _p, all_probs = model.detect_language(audio, vad_filter=args.vad_filter)
            language, language_prob = choose_language(all_probs, languages, args.switch_threshold)
        result = transcribe(audio, language)
        if language == "ja" and "en" in languages and looks_like_katakana_english(result["text"]):
            # なまりの強い英語が日本語扱いになったときだけ、英語でもう一度文字にする
            result = transcribe(audio, "en")
            language, result["retried_as"] = "en", "en"
        return {**result, "language": language, "language_prob": round(language_prob, 3)}

    def transcribe(audio, language: str) -> dict:
        segments, _info = model.transcribe(
            audio, language=language, vad_filter=args.vad_filter,
            beam_size=5, condition_on_previous_text=False,
        )
        parts, probs, logprobs = [], [], []
        for segment in segments:
            parts.append(segment.text)
            probs.append(max(0.0, min(1.0, 1.0 - segment.no_speech_prob)))
            logprobs.append(segment.avg_logprob)
        return {"text": "".join(parts).strip(), "confidence": sum(probs) / len(probs) if probs else 0.0,
                "avg_logprob": round(sum(logprobs) / len(logprobs), 3) if logprobs else None}

    # GPU が使えるかをここで確かめ、だめなら CPU に切り替える
    if args.device == "cuda":
        try:
            import numpy as np
            list(model.transcribe(np.zeros(16000, dtype=np.float32), language=languages[0])[0])
        except Exception as error:
            print(f"[stt_worker] GPU で動かせないため CPU(int8) に切り替えます: {error}", file=sys.stderr, flush=True)
            model = WhisperModel(args.model, device="cpu", compute_type="int8")
            args.device = "cpu"
    send({"id": "ready", "type": "ready", "device": args.device})

    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            send({"id": request["id"], "type": "completed", **run(request["audioPath"])})
        except Exception as error:  # keep the worker alive for the next request
            traceback.print_exc(file=sys.stderr)
            send({"id": request.get("id", "unknown"), "type": "error", "message": str(error)})


if __name__ == "__main__":
    main()
