"""faster-whisper worker (small / ja / VAD, beam 5).

Runs as a subprocess of r1_chat.py (same pixi env by default).
Protocol: one JSON per line on stdin {"id", "audioPath"} -> one JSON per line on stdout
{"id", "type": "completed"|"error", "text", "confidence"}. Logs go to stderr.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback


def send(message: dict) -> None:
    print(json.dumps(message, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--vad-filter", action="store_true")
    parser.add_argument("--dll-dir", action="append", default=[],
                        help="cuBLAS 12 / cuDNN 9 の DLL があるフォルダ（Windows の GPU 用）")
    args = parser.parse_args()

    for d in args.dll_dir:
        if os.path.isdir(d):
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]

    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    def run(path: str) -> dict:
        segments, _info = model.transcribe(
            path, language=args.language, vad_filter=args.vad_filter,
            beam_size=5, condition_on_previous_text=False,
        )
        parts, probs = [], []
        for segment in segments:
            parts.append(segment.text)
            probs.append(max(0.0, min(1.0, 1.0 - segment.no_speech_prob)))
        return {"text": "".join(parts).strip(), "confidence": sum(probs) / len(probs) if probs else 0.0}

    # GPU が使えるかをここで確かめ、だめなら CPU に切り替える
    if args.device == "cuda":
        try:
            import numpy as np
            list(model.transcribe(np.zeros(16000, dtype=np.float32), language=args.language)[0])
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
