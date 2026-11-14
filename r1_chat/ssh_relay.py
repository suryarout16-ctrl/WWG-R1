"""SSH のコマンド実行で、Z840 の中だけで開いている Strata（127.0.0.1:8081）を呼ぶ。

Z840 はポート転送（トンネル）を禁止しているので、SSH で Z840 上に小さな中継プログラムを1つ起動したままにし、
標準入出力で 1 行 1 JSON の依頼・返事をやり取りする。Z840 の設定は変えない。

  依頼: {"op": "models"} / {"op": "chat", "body": {... OpenAI 形式、stream: true ...}}
  返事: {"ok": true, "data": ...} / {"ok": true, "content": "本文"} / {"ok": false, "status": 400, "error": "..."}
"""
from __future__ import annotations

import base64
import json
import os
import queue
import subprocess
import threading
from pathlib import Path

REMOTE = r'''
import sys, json, urllib.request, urllib.error
BASE = "http://127.0.0.1:%PORT%"
def out(o):
    sys.stdout.write(json.dumps(o, ensure_ascii=False) + "\n"); sys.stdout.flush()
out({"ok": True, "type": "ready"})
for line in sys.stdin:
    try:
        req = json.loads(line)
        if req.get("op") == "models":
            out({"ok": True, "data": json.loads(urllib.request.urlopen(BASE + "/v1/models", timeout=5).read())})
            continue
        r = urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions",
            json.dumps(req["body"]).encode(), {"Content-Type": "application/json"}), timeout=300)
        content = ""
        for raw in r:
            s = raw.decode("utf-8", "replace").strip()
            if not s.startswith("data:"):
                continue
            d = s[5:].strip()
            if d == "[DONE]":
                break
            c = json.loads(d)
            if "error" in c:
                raise RuntimeError(str(c["error"])[:200])
            for ch in c.get("choices", []):
                content += (ch.get("delta") or {}).get("content") or ""
        out({"ok": True, "content": content})
    except urllib.error.HTTPError as e:
        out({"ok": False, "status": e.code, "error": e.read()[:300].decode("utf-8", "replace")})
    except Exception as e:
        out({"ok": False, "error": (type(e).__name__ + ": " + str(e))[:300]})
'''


class RelayError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class SshRelay:
    def __init__(self, target: str, key_dir: str = "", host_key_alias: str = "", port: int = 8081):
        self.target, self.port = target, port
        ssh = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "OpenSSH" / "ssh.exe"
        self.cmd = [str(ssh) if ssh.exists() else "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                    "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3"]
        if key_dir:
            self.cmd += ["-i", str(Path(key_dir) / "id_ed25519"),
                         "-o", f"UserKnownHostsFile={Path(key_dir) / 'known_hosts'}"]
        if host_key_alias:
            self.cmd += ["-o", f"HostKeyAlias={host_key_alias}"]
        script = base64.b64encode(REMOTE.replace("%PORT%", str(port)).encode()).decode()
        self.cmd += [target, f"python3 -I -u -c \"import base64;exec(base64.b64decode('{script}'))\""]
        self.proc = None
        self.lines: queue.Queue = queue.Queue()
        self.lock = threading.Lock()

    def _start(self) -> None:
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, encoding="utf-8", errors="replace",
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.lines = queue.Queue()
        proc, lines = self.proc, self.lines

        def pump():
            for line in proc.stdout:
                lines.put(line)
            lines.put(None)
        threading.Thread(target=pump, daemon=True).start()
        first = self._read(20.0)
        if not first.get("ok"):
            raise RelayError(f"Z840 の中継を起動できません: {first}")

    def _read(self, timeout: float) -> dict:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise RelayError(f"Z840 から {timeout:.0f} 秒応答がありません")
        if line is None:
            err = self.proc.stderr.read()[:300] if self.proc and self.proc.stderr else ""
            self.close()
            raise RelayError(f"SSH が切れました: {err.strip()}")
        return json.loads(line)

    def request(self, payload: dict, timeout: float) -> dict:
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            try:
                self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                self.proc.stdin.flush()
            except OSError as error:
                self.close()
                raise RelayError(f"SSH に書き込めません: {error}")
            reply = self._read(timeout)
        if not reply.get("ok"):
            raise RelayError(reply.get("error", "不明なエラー"), reply.get("status"))
        return reply

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        self.proc = None
