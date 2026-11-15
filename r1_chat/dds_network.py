"""DDSの送信元IPv4を、このPythonプロセスのSDK設定だけで指定する。"""
from __future__ import annotations

from contextlib import nullcontext
import importlib
import ipaddress


_INTERFACE_MARKER = 'name="$__IF_NAME__$"'


def configure_dds_address(address, channel_module=None) -> str | None:
    """初期化前のSDKテンプレートをaddress属性へ変える。SDKファイルやNIC設定は変更しない。

    Noneと空文字は無指定で、SDKをimportしない。インターフェース名とIPv4を
    同時指定するとWindowsの同名・複数IPv4の選択に失敗するため、addressだけを使う。
    """
    if address is None:
        return None
    if not isinstance(address, str):
        raise ValueError("DDSの接続元はIPv4アドレスの文字列で指定してください")
    if address == "":
        return None
    try:
        parsed = ipaddress.IPv4Address(address)
    except ipaddress.AddressValueError:
        raise ValueError("DDSの接続元は有効なIPv4アドレスで指定してください") from None
    if parsed.is_unspecified or parsed.is_multicast or parsed == ipaddress.IPv4Address("255.255.255.255"):
        raise ValueError("DDSの接続元に無指定・マルチキャスト・ブロードキャストは使えません")
    selected = str(parsed)
    channel = channel_module if channel_module is not None else importlib.import_module("unitree_sdk2py.core.channel")
    factory = getattr(channel, "ChannelFactory", None)
    if factory is None:
        raise RuntimeError("SDKのChannelFactoryを確認できません")
    # SDK自身のInitと同じロックで、確認後に初期化が割り込むのを防ぐ。
    lock = getattr(factory, "_ChannelFactory__init_lock", None)
    with lock if lock is not None else nullcontext():
        if getattr(factory, "_ChannelFactory__initialized", False):
            raise RuntimeError("DDS初期化後は接続元を変更できません。新しいプロセスで起動してください")
        original = getattr(channel, "ChannelConfigHasInterface", None)
        if not isinstance(original, str) or original.count(_INTERFACE_MARKER) != 1:
            raise RuntimeError("SDKのインターフェース設定形式を確認できないため変更しません")
        updated = original.replace(_INTERFACE_MARKER, f'address="{selected}"', 1)
        channel.ChannelConfigHasInterface = updated
    return selected
