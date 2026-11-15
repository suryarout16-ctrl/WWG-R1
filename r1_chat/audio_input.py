"""入力マイクの名前を保ち、Windowsでは同じ機器のWASAPI入力を選ぶ。"""
from __future__ import annotations

import importlib
import re
import sys
import threading
import unicodedata


_AUDIO_REFRESH_LOCK = threading.Lock()


def refresh_audio_devices(sounddevice_module=None) -> None:
    """閉じた入力の再接続前に、このプロセスのPortAudio機器情報を再列挙する。

    呼出し側が入力のライフサイクルを排他し、所有ストリームをすべて閉じてから使う。
    NIC・Bluetooth・Windowsの既定デバイスを変更せず、ストリームも開かない。
    private関数の契約は同梱sounddevice.pyの_initialize/_terminateで確認している。
    """
    sd = sounddevice_module if sounddevice_module is not None else importlib.import_module("sounddevice")
    initialize, terminate = getattr(sd, "_initialize", None), getattr(sd, "_terminate", None)
    if not callable(initialize) or not callable(terminate):
        raise RuntimeError("この音声ライブラリでは入力機器情報を安全に更新できません")
    with _AUDIO_REFRESH_LOCK:
        count = getattr(sd, "_initialized", None)
        if isinstance(count, bool) or not isinstance(count, int) or count not in (0, 1):
            raise RuntimeError("PortAudioの初期化状態を確認できないため更新しません")
        convenience = getattr(sd, "_last_callback", None)
        if convenience is not None and getattr(getattr(convenience, "stream", None), "active", False):
            raise RuntimeError("別の音声ストリームが使用中のため機器情報を更新しません")
        if count == 1:
            try:
                terminate()
            except Exception:
                raise RuntimeError("入力機器情報の更新を開始できませんでした") from None
        try:
            initialize()
        except Exception:
            # _initializeは成功時だけ参照数を増やす。0のままなら一度だけ復旧を試す。
            if getattr(sd, "_initialized", None) == 0:
                try:
                    initialize()
                except Exception:
                    pass
            raise RuntimeError("入力機器情報を更新できませんでした。接続待ちで再試行します") from None
        if getattr(sd, "_initialized", None) != 1:
            raise RuntimeError("更新後のPortAudio初期化状態を確認できません")


def _normal(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def _friendly_key(name):
    """Windows Bluetoothの未展開リソース名だけを、同一の機器名へ戻す。"""
    value = _normal(name)
    if "bthhfenum.sys" not in value or "@" not in value or ";" not in value:
        return value
    prefix = value.split("(", 1)[0].strip()
    suffix = value.rsplit(";", 1)[-1].strip()
    if not prefix or not suffix.startswith("("):
        return value
    depth = 0
    for index, char in enumerate(suffix):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                model = suffix[1:index].strip()
                return f"{prefix} ({model})" if model else value
    return value


def resolve_input_device(requested, sounddevice_module=None):
    """(index, metadata)を返す。列挙だけでストリームは開かない。

    数字指定はその入力の名前を基準とし、別の名前のマイクには切り替えない。
    名前指定が複数の物理機器へ一致した場合は推測せず拒否する。
    """
    if isinstance(requested, bool) or not isinstance(requested, (str, int, type(None))):
        raise ValueError("入力マイクは名前または番号で指定してください")
    if isinstance(requested, str):
        requested = requested.strip()
        if not requested:
            raise ValueError("入力マイク名が空です")
        if requested.isascii() and requested.isdigit():
            requested = int(requested)
    if isinstance(requested, int) and requested < 0:
        raise ValueError("入力マイク番号が不正です")
    sd = sounddevice_module if sounddevice_module is not None else importlib.import_module("sounddevice")
    devices = list(sd.query_devices())
    hosts = list(sd.query_hostapis())
    inputs = []
    for position, row in enumerate(devices):
        if row.get("max_input_channels", 0) <= 0:
            continue
        index = row.get("index", position)
        name = row.get("name")
        host_index = row.get("hostapi")
        if (isinstance(index, bool) or not isinstance(index, int) or index < 0 or not isinstance(name, str)
                or isinstance(host_index, bool) or not isinstance(host_index, int)
                or not 0 <= host_index < len(hosts)):
            raise RuntimeError("入力マイクの情報を確認できません")
        inputs.append({"index": index, "name": name, "hostapi": hosts[host_index]["name"],
                       "rate": row.get("default_samplerate"), "key": _friendly_key(name)})
    if requested is None:
        default = sd.query_devices(kind="input")
        requested = default.get("index")
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 0:
            raise ValueError("Windowsの既定入力マイクを確認できません")

    windows = sys.platform == "win32"
    if isinstance(requested, int):
        candidates = [row for row in inputs if row["index"] == requested]
        if len(candidates) != 1:
            raise ValueError("指定した番号の入力マイクがありません")
        selected = candidates[0]
        if windows and "wasapi" not in selected["hostapi"].casefold():
            preferred = [row for row in inputs if row["key"] == selected["key"]
                         and "wasapi" in row["hostapi"].casefold()]
            if len(preferred) > 1:
                raise ValueError("同じ名前のWASAPI入力が複数あります。番号で選び直してください")
            if preferred:
                selected = preferred[0]
    else:
        query = _friendly_key(requested)
        candidates = [row for row in inputs if query in _normal(row["name"]) or query in row["key"]]
        exact = [row for row in candidates if query in (_normal(row["name"]), row["key"])]
        if exact:
            candidates = exact
        if not candidates:
            raise ValueError("指定した名前の入力マイクがありません")
        if len({row["key"] for row in candidates}) != 1:
            raise ValueError("入力マイク名が複数の機器に一致します。詳しい名前で指定してください")

        def rank(row):
            if not windows:
                return 0
            name = row["hostapi"].casefold()
            return 0 if "wasapi" in name else 1 if "directsound" in name else 2 if name == "mme" else 3 if "wdm" in name else 4

        best = min(rank(row) for row in candidates)
        preferred = [row for row in candidates if rank(row) == best]
        if len(preferred) != 1:
            raise ValueError("同じ名前の入力マイクが複数あります。番号で指定してください")
        selected = preferred[0]
    metadata = {key: selected[key] for key in ("index", "name", "hostapi", "rate")}
    return selected["index"], metadata
