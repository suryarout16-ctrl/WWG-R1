"""R1 の DDS 型定義を保存し、R1 が無くても録画（r1_snapshot.py の raw/*.pkl）を読めるようにする。

  保存（R1 が必要・約 10 秒、読み取りだけ）:
    pixi run python local_tools/r1_types.py dump -i Ethernet --out logs/r1_snapshot_<日時>
  確認（R1 不要）:
    pixi run python local_tools/r1_types.py check logs/r1_snapshot_<日時>

型は XTypes の TypeObject（バイト列）で保存し、読むときに CycloneDDS の XTInterpreter で Python の型に戻す。
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from collections import deque
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from cyclonedds.idl._typesupport.DDS.XTypes._ddsi_xt_type_object import TypeIdentifier, TypeObject
from cyclonedds.idl._xt_builder import XTInterpreter, XTTypeIdScanner

TYPES_FILE = "types.pkl"


def _ser_tid(tid) -> bytes:
    return tid.serialize(use_version_2=True)


def _de_tid(b: bytes):
    return TypeIdentifier.deserialize(b)   # ヘッダ付きなので版はヘッダから読む


def dump(interface: str, out: Path, discover_s: float = 5.0) -> None:
    from cyclonedds import _clayer as cl
    from cyclonedds.builtin import BuiltinDataReader, BuiltinTopicDcpsPublication
    from unitree_sdk2py.core.channel import ChannelFactory, ChannelFactoryInitialize

    ChannelFactoryInitialize(0, interface)
    dp = ChannelFactory()._ChannelFactory__participant
    reader = BuiltinDataReader(dp, BuiltinTopicDcpsPublication)
    time.sleep(discover_s)
    topics = {}
    for s in reader.take(N=10000):
        if s.type_id is not None and not s.topic_name.startswith("DCPS"):
            topics.setdefault(s.topic_name, (s.type_name, s.type_id))
    saved, objects = {}, {}
    for name, (type_name, tid) in sorted(topics.items()):
        todo, deps = deque([tid]), []
        try:
            while todo:
                t = todo.pop()
                key = _ser_tid(t)
                if key in deps:
                    continue
                if key not in objects:
                    ret = cl.ddspy_get_typeobj(dp._ref, t.serialize(use_version_2=True)[4:], 3_000_000_000)
                    if isinstance(ret, int):
                        raise RuntimeError(f"TypeObject を取れない ({ret})")
                    objects[key] = ret
                obj = TypeObject.deserialize(objects[key], has_header=False, use_version_2=True)
                deps.append(key)
                todo.extend(XTTypeIdScanner.find_all_typeids(obj))
            saved[name] = {"type_name": type_name, "type_id": _ser_tid(tid), "deps": deps}
            print(f"  + {name} ({type_name}, 依存 {len(deps)})")
        except Exception as error:
            print(f"  - {name}: {type(error).__name__}: {error}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / TYPES_FILE, "wb") as fh:
        pickle.dump({"topics": saved, "objects": objects}, fh)
    print(f"保存: {out / TYPES_FILE}（トピック {len(saved)}、TypeObject {len(objects)}）")


def load(folder: Path) -> dict:
    """{トピック名: Python の型} を返す（R1 不要）。"""
    data = pickle.load(open(folder / TYPES_FILE, "rb"))
    objects, out = data["objects"], {}
    for name, info in data["topics"].items():
        typemap = {_de_tid(k): TypeObject.deserialize(objects[k], has_header=False, use_version_2=True)
                   for k in info["deps"]}
        out[name] = XTInterpreter.xt_to_class(_de_tid(info["type_id"]), typemap)[0]
    return out


def read_raw(path: Path):
    with open(path, "rb") as fh:
        while True:
            try:
                yield pickle.load(fh)
            except EOFError:
                return


def check(folder: Path) -> None:
    types = load(folder)
    ok = ng = 0
    for raw in sorted((folder / "raw").glob("*.pkl")):
        topic = next((n for n in types if n.replace("/", "_") + ".pkl" == raw.name), None)
        first = next(read_raw(raw), None)
        if topic is None or first is None:
            continue
        try:
            sample = types[topic].deserialize(first[1])
            ok += 1
            print(f"  OK {topic}: {str(sample)[:100]}")
        except Exception as error:
            ng += 1
            print(f"  NG {topic}: {type(error).__name__}: {error}")
    print(f"読めた {ok} / 読めない {ng}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("-i", "--interface", default="Ethernet")
    d.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("folder")
    a = ap.parse_args()
    if a.cmd == "dump":
        dump(a.interface, Path(a.out))
    else:
        check(Path(a.folder))
