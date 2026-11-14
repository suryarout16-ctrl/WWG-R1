"""OpenAI の API キーを Windows 資格情報マネージャーに保存・読み出す（このPCの中だけ。DPAPIで暗号化される）。

キーはリポジトリにもファイルにも書かない。r1_chat.py は 環境変数 OPENAI_API_KEY → 資格情報マネージャー の順で探す。

  pixi run python r1_chat/credentials.py status              # 保存済みかだけ表示（キーは表示しない）
  pixi run python r1_chat/credentials.py set                 # キーを入力して保存（入力は画面に出ない）
  pixi run python r1_chat/credentials.py copy <元のターゲット名>  # 別アプリが保存したキーを複製
  pixi run python r1_chat/credentials.py delete
"""
from __future__ import annotations

import sys

DEFAULT_TARGET = "WWG-R1/OpenAI"
CRED_TYPE_GENERIC = 1
CRED_PERSIST_LOCAL_MACHINE = 2
ERROR_NOT_FOUND = 1168


def _api():
    import ctypes
    from ctypes import wintypes

    class CREDENTIAL(ctypes.Structure):
        _fields_ = [("Flags", wintypes.DWORD), ("Type", wintypes.DWORD), ("TargetName", wintypes.LPWSTR),
                    ("Comment", wintypes.LPWSTR), ("LastWritten", wintypes.FILETIME),
                    ("CredentialBlobSize", wintypes.DWORD), ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
                    ("Persist", wintypes.DWORD), ("AttributeCount", wintypes.DWORD), ("Attributes", ctypes.c_void_p),
                    ("TargetAlias", wintypes.LPWSTR), ("UserName", wintypes.LPWSTR)]

    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    advapi.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(ctypes.POINTER(CREDENTIAL))]
    advapi.CredReadW.restype = wintypes.BOOL
    advapi.CredWriteW.argtypes = [ctypes.POINTER(CREDENTIAL), wintypes.DWORD]
    advapi.CredWriteW.restype = wintypes.BOOL
    advapi.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
    advapi.CredDeleteW.restype = wintypes.BOOL
    advapi.CredFree.argtypes = [ctypes.c_void_p]
    return ctypes, CREDENTIAL, advapi


def get_secret(target: str = DEFAULT_TARGET) -> str | None:
    if sys.platform != "win32":
        return None
    ctypes, CREDENTIAL, advapi = _api()
    pointer = ctypes.POINTER(CREDENTIAL)()
    if not advapi.CredReadW(target, CRED_TYPE_GENERIC, 0, ctypes.byref(pointer)):
        if ctypes.get_last_error() == ERROR_NOT_FOUND:
            return None
        raise OSError(f"CredRead failed: {ctypes.get_last_error()}")
    try:
        cred = pointer.contents
        raw = ctypes.string_at(cred.CredentialBlob, cred.CredentialBlobSize)
        return raw.decode("utf-16-le").strip() or None
    finally:
        advapi.CredFree(pointer)


def set_secret(secret: str, target: str = DEFAULT_TARGET) -> None:
    secret = secret.strip()
    if not secret:
        raise ValueError("空のキーは保存できません")
    ctypes, CREDENTIAL, advapi = _api()
    data = secret.encode("utf-16-le")
    blob = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    cred = CREDENTIAL(Type=CRED_TYPE_GENERIC, TargetName=target, CredentialBlobSize=len(data),
                      CredentialBlob=ctypes.cast(blob, ctypes.POINTER(ctypes.c_ubyte)),
                      Persist=CRED_PERSIST_LOCAL_MACHINE, UserName="WWG-R1")
    if not advapi.CredWriteW(ctypes.byref(cred), 0):
        raise OSError(f"CredWrite failed: {ctypes.get_last_error()}")


def delete_secret(target: str = DEFAULT_TARGET) -> None:
    ctypes, _, advapi = _api()
    if not advapi.CredDeleteW(target, CRED_TYPE_GENERIC, 0) and ctypes.get_last_error() != ERROR_NOT_FOUND:
        raise OSError(f"CredDelete failed: {ctypes.get_last_error()}")


def describe(secret: str | None) -> str:
    return "未保存" if not secret else f"保存済み（{len(secret)}文字、先頭 {secret[:3]}…）"


def main() -> None:
    if sys.platform != "win32":
        sys.exit("Windows 専用です。他の OS では環境変数 OPENAI_API_KEY を使ってください")
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    if action == "status":
        print(f"{DEFAULT_TARGET}: {describe(get_secret())}")
    elif action == "set":
        import getpass
        set_secret(getpass.getpass("OpenAI API キー（入力は表示されません）: "))
        print(f"{DEFAULT_TARGET}: {describe(get_secret())}")
    elif action == "copy" and len(sys.argv) > 2:
        secret = get_secret(sys.argv[2])
        if not secret:
            sys.exit(f"{sys.argv[2]} にキーが見つかりません")
        set_secret(secret)
        print(f"{sys.argv[2]} → {DEFAULT_TARGET}: {describe(get_secret())}")
    elif action == "delete":
        delete_secret()
        print(f"{DEFAULT_TARGET}: 削除しました")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
