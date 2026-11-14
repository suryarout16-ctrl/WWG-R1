"""R1 の「読み取りだけ」のスナップショット。R1 には何も動かす命令を送らない。

  1. DDS で見えるトピック（誰が何を出しているか、型名、QoS）
  2. 各トピックの中身を数秒間録画（CDR の生データ＋JSON の見本）→ 自宅で再生用
  3. 読み取り専用のサービス呼び出し：FSM、腕の動作一覧、音量、サービス一覧、モード確認

使い方（リポジトリのルート）:
  pixi run python local_tools/r1_snapshot.py -i Ethernet --seconds 15
出力: logs/r1_snapshot_<日時>/
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "r1_motion"))
sys.stdout.reconfigure(encoding="utf-8")

from cyclonedds.builtin import BuiltinDataReader, BuiltinTopicDcpsParticipant, BuiltinTopicDcpsPublication, \
    BuiltinTopicDcpsSubscription
from cyclonedds.core import Listener
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from cyclonedds.util import duration
from unitree_sdk2py.core.channel import ChannelFactory, ChannelFactoryInitialize
from unitree_sdk2py.rpc.client import Client

MAX_BYTES_PER_TOPIC = 200 * 1024 * 1024
MAX_JSON_SAMPLES = 50


def to_jsonable(obj, depth=0):
    if depth > 12:
        return "..."
    if dataclasses.is_dataclass(obj):
        return {f.name: to_jsonable(getattr(obj, f.name), depth + 1) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x, depth + 1) for x in obj]
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    try:
        return to_jsonable(vars(obj), depth + 1)
    except TypeError:
        return repr(obj)


def known_types() -> dict:
    """unitree_sdk2py に入っている型を、DDS の型名（a::b::C）で引けるようにする。"""
    import importlib
    import pkgutil
    import unitree_sdk2py.idl as idl
    out = {}
    for mod in pkgutil.walk_packages(idl.__path__, idl.__name__ + "."):
        try:
            m = importlib.import_module(mod.name)
        except Exception:
            continue
        for v in vars(m).values():
            name = getattr(v, "__idl_typename__", None)
            if isinstance(v, type) and name:
                out[name.replace(".", "::")] = v
    return out


def qos_text(q):
    try:
        return [repr(p) for p in q] if q is not None else None
    except Exception:
        return repr(q)


def discover(dp, seconds: float) -> dict:
    readers = {"participants": BuiltinDataReader(dp, BuiltinTopicDcpsParticipant),
               "publications": BuiltinDataReader(dp, BuiltinTopicDcpsPublication),
               "subscriptions": BuiltinDataReader(dp, BuiltinTopicDcpsSubscription)}
    time.sleep(seconds)
    found = {k: {} for k in readers}
    for kind, r in readers.items():
        for s in r.take(N=10000):
            key = str(s.key)
            if kind == "participants":
                found[kind][key] = {"qos": qos_text(s.qos)}
            else:
                found[kind][key] = {"topic": s.topic_name, "type": s.type_name, "participant": str(s.participant_key),
                                    "qos": qos_text(s.qos), "type_id": s.type_id}
    return found


def record(dp, pubs: dict, seconds: float, out: Path, log) -> dict:
    """出ているトピックを全部購読して録画。型は動的に取得、駄目なら SDK の型。"""
    try:
        from cyclonedds.dynamic import get_types_for_typeid
    except ImportError:
        get_types_for_typeid = None
    sdk_types = known_types()
    topics = {}
    for p in pubs.values():
        topics.setdefault(p["topic"], p)
    readers, stats = {}, {}
    lock = threading.Lock()
    for name, p in sorted(topics.items()):
        if name.startswith("DCPS"):
            continue
        typ, how = None, ""
        if get_types_for_typeid and p.get("type_id") is not None:
            try:
                typ, _ = get_types_for_typeid(dp, p["type_id"], duration(seconds=3))
                how = "dynamic"
            except Exception as error:
                how = f"dynamic failed: {type(error).__name__}"
        if typ is None and p["type"] in sdk_types:
            typ, how = sdk_types[p["type"]], "sdk"
        st = {"type": p["type"], "how": how, "count": 0, "bytes": 0, "first": None, "last": None, "samples": [],
              "error": ""}
        stats[name] = st
        if typ is None:
            st["error"] = "型が分からないので録画できない"
            log(f"  - {name} ({p['type']}): {st['error']} [{how}]")
            continue
        raw_path = out / "raw" / (name.replace("/", "_") + ".pkl")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(raw_path, "wb")

        def on_data(reader, name=name, st=st, fh=fh):
            for s in reader.take(N=1000):
                now = time.time()
                with lock:
                    if st["bytes"] > MAX_BYTES_PER_TOPIC:
                        continue
                    try:
                        data = s.serialize()
                    except Exception:
                        data = b""
                    pickle.dump((now, data), fh)
                    st["count"] += 1
                    st["bytes"] += len(data)
                    st["first"] = st["first"] or now
                    st["last"] = now
                    if len(st["samples"]) < MAX_JSON_SAMPLES and (st["count"] == 1 or st["count"] % 25 == 0):
                        st["samples"].append({"t": now, "data": to_jsonable(s)})
        try:
            topic = Topic(dp, name, typ)
            readers[name] = (DataReader(dp, topic, listener=Listener(on_data_available=on_data)), fh)
            log(f"  + {name} ({p['type']}) [{how}]")
        except Exception as error:
            st["error"] = f"{type(error).__name__}: {error}"
            fh.close()
            log(f"  - {name}: {st['error']}")
    time.sleep(seconds)
    for reader, fh in readers.values():
        try:
            reader.set_listener(None)
        except Exception:
            pass
        with lock:
            fh.close()
    for st in stats.values():
        span = (st["last"] - st["first"]) if st["first"] and st["last"] else 0
        st["hz"] = round(st["count"] / span, 1) if span > 0 else None
    return stats


class ReadOnly(Client):
    def __init__(self, service, version, apis):
        super().__init__(service, False)
        self._ver, self._apis = version, apis

    def Init(self):
        self._SetApiVerson(self._ver)
        for a in self._apis:
            self._RegistApi(a, 0)


def rpc_calls(log) -> list:
    """読み取り専用の API だけを呼ぶ（動かす API は呼ばない）。"""
    calls = [
        ("sport", "1.0.0.0", 7001, "{}", "GetFsmId"),
        ("sport", "1.0.0.0", 7002, "{}", "GetFsmMode"),
        ("arm", "1.0.0.0", 7107, "{}", "GetActionList"),
        ("voice", "1.0.0.0", 1005, "{}", "GetVolume (G1 audio)"),
        ("motion_switcher", "1.0.0.1", 1001, "{}", "CheckMode"),
        ("motion_switcher", "1.0.0.1", 1005, "{}", "GetSilent"),
        ("robot_state", "1.0.0.1", 1003, "{}", "ServiceList"),
    ]
    results = []
    for service, ver, api, param, label in calls:
        c = ReadOnly(service, ver, [api])
        c.SetTimeout(5.0)
        c.Init()
        t0 = time.time()
        try:
            code, data = c._Call(api, param)
        except Exception as error:
            code, data = None, f"{type(error).__name__}: {error}"
        try:
            parsed = json.loads(data) if data else data
        except (TypeError, ValueError):
            parsed = data
        results.append({"service": service, "api": api, "label": label, "code": code, "data": parsed,
                        "sec": round(time.time() - t0, 2)})
        log(f"  {service}/{api} {label}: code={code} {str(data)[:150]}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--interface", default="Ethernet")
    ap.add_argument("--seconds", type=float, default=15.0, help="録画する秒数")
    ap.add_argument("--discover", type=float, default=5.0, help="トピック探索の秒数")
    args = ap.parse_args()

    out = ROOT / "logs" / time.strftime("r1_snapshot_%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    log_fh = open(out / "snapshot.log", "w", encoding="utf-8")

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    ChannelFactoryInitialize(0, args.interface)
    dp = ChannelFactory()._ChannelFactory__participant
    log(f"出力先: {out}")

    log("① 読み取り専用のサービス呼び出し")
    rpc = rpc_calls(log)
    (out / "rpc.json").write_text(json.dumps(rpc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    log(f"② DDS の探索（{args.discover:.0f} 秒）")
    found = discover(dp, args.discover)
    pubs = found["publications"]
    log(f"  参加者 {len(found['participants'])} / 出している {len(pubs)} / 受けている {len(found['subscriptions'])}")
    (out / "discovery.json").write_text(json.dumps(found, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    log(f"③ 録画（{args.seconds:.0f} 秒）")
    stats = record(dp, pubs, args.seconds, out, log)
    (out / "topics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for name, st in sorted(stats.items()):
        log(f"  {name}: {st['count']} 件, {st['hz']} Hz, {st['bytes'] / 1e6:.1f} MB {st['error']}")
    log("完了")


if __name__ == "__main__":
    main()
