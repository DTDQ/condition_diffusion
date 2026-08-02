#!/usr/bin/env python3
"""
Self-contained Tonghuashun iFinD HTTP API client + CLI for the quant-data skill.

Calls https://quantapi.51ifind.com/api/v1. Token handling is automatic:
  - access_token  (short-lived)   : sent in the `access_token` request header.
  - refresh_token (long-lived)    : used to obtain a fresh access_token.
  - When the server returns errorcode -1302 or -1303 (token expired), the
    client force-refreshes the access_token and retries the request once.
  - On a successful refresh the new access_token is written back to ifind.env
    so subsequent processes (and other tools) reuse it.

Token config (chmod 600) is read from (first found):
  1. ~/.zcode/skills/quant-data/ifind.env
  2. ~/.ifind-http.env
  3. ./ifind.env, ./config/ifind.env, ./.env
Env vars IFIND_REFRESH_TOKEN / IFIND_ACCESS_TOKEN / IFIND_BASE_URL override all.

When refresh_token itself eventually expires (currently valid until 2026-12-31),
obtain a new one from the iFinD "超级命令" (Super Command) account page and put
it in ifind.env. See SKILL.md "Token refresh".

Usage:
  python3 ifind.py token                       # force-refresh + print access_token
  python3 ifind.py smoke [--codes 300033.SZ]   # connectivity check
  python3 ifind.py realtime --codes 300033.SZ [--indicators latest]
  python3 ifind.py history --codes 000001.SZ,600000.SH --indicators open,high,low,close \
                              --startdate 2021-07-05 --enddate 2022-07-05
  python3 ifind.py edb --indicators M001620326 --startdate 2020-01-01 --enddate 2022-12-31
  python3 ifind.py post ENDPOINT --json '{"...":"..."}'   # generic POST

Output is JSON (pretty by default; --compact for one line).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_HERE = Path(__file__).resolve().parent
_ENV_FILE = _HERE / "ifind.env"

# errorcodes that mean "access_token expired -> refresh and retry".
TOKEN_EXPIRED_CODES = {-1302, -1303}

_CONFIG_PATHS = (
    _ENV_FILE,                       # skill's own env (preferred)
    Path("~/.ifind-http.env").expanduser(),
    Path("ifind.env"),
    Path("config/ifind.env"),
    Path(".env"),
)


class IFindError(Exception):
    """Raised for API/config errors."""


def _load_env(path: Optional[Path] = None) -> Dict[str, str]:
    """Load KEY=VALUE pairs from a dotenv file (without external deps)."""
    cfg: Dict[str, str] = {}
    candidates = [path] if path else list(_CONFIG_PATHS)
    for p in candidates:
        if p and p.exists():
            for raw in p.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k:
                    cfg[k] = v
            if path:  # explicit path only
                break
    # env vars win
    for k in ("IFIND_REFRESH_TOKEN", "IFIND_ACCESS_TOKEN", "IFIND_BASE_URL"):
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    return cfg


def _save_access_token(token: str) -> None:
    """Write the refreshed access_token back to the skill's ifind.env."""
    if not _ENV_FILE.exists():
        return
    lines = _ENV_FILE.read_text(encoding="utf-8").splitlines()
    out, seen = [], False
    for line in lines:
        if line.strip().startswith("IFIND_ACCESS_TOKEN="):
            out.append(f"IFIND_ACCESS_TOKEN={token}")
            seen = True
        else:
            out.append(line)
    if not seen:
        out.append(f"IFIND_ACCESS_TOKEN={token}")
    try:
        _ENV_FILE.write_text("\n".join(out) + "\n", encoding="utf-8")
        os.chmod(_ENV_FILE, 0o600)
    except OSError:
        pass  # non-fatal: in-memory token still works for this process


class IFindClient:
    def __init__(self, dotenv_path: Optional[Path] = None, timeout: int = 30) -> None:
        import requests  # imported lazily so `dict`-style help works without it
        self.cfg = _load_env(dotenv_path)
        self.requests = requests
        self.refresh_token = self.cfg.get("IFIND_REFRESH_TOKEN")
        self.access_token = self.cfg.get("IFIND_ACCESS_TOKEN")
        self.base_url = (self.cfg.get("IFIND_BASE_URL")
                         or "https://quantapi.51ifind.com/api/v1").rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _require(self) -> None:
        if not self.refresh_token:
            raise IFindError(
                "Missing IFIND_REFRESH_TOKEN. Get it from the iFinD Super Command "
                "account page and put it in " + str(_ENV_FILE) + ".")

    def get_access_token(self, force: bool = False) -> str:
        """Return a usable access_token, refreshing from refresh_token if needed."""
        if self.access_token and not force:
            return self.access_token
        self._require()
        resp = self.session.post(
            f"{self.base_url}/get_access_token",
            headers={"Content-Type": "application/json", "refresh_token": self.refresh_token},
            timeout=self.timeout,
        )
        data = self._decode(resp)
        token = (data.get("data") or {}).get("access_token")
        if not token:
            raise IFindError(f"refresh failed: errorcode={data.get('errorcode')} "
                             f"errmsg={data.get('errmsg')}")
        self.access_token = token
        _save_access_token(token)  # persist for other processes
        return token

    def post(self, endpoint: str, payload: Optional[Dict[str, Any]] = None,
             retry_token: bool = True) -> Dict[str, Any]:
        endpoint = endpoint.strip("/")
        token = self.get_access_token()
        resp = self.session.post(
            f"{self.base_url}/{endpoint}",
            json=payload or {},
            headers={"Content-Type": "application/json", "access_token": token},
            timeout=self.timeout,
        )
        data = self._decode(resp)
        code = data.get("errorcode", 0)
        if retry_token and code in TOKEN_EXPIRED_CODES:
            self.get_access_token(force=True)
            return self.post(endpoint, payload, retry_token=False)
        if code not in (0, None):
            raise IFindError(f"errorcode={code} errmsg={data.get('errmsg','')} "
                             f"(endpoint={endpoint})")
        return data

    @staticmethod
    def _decode(resp) -> Dict[str, Any]:
        try:
            return resp.json()
        except ValueError as exc:
            raise IFindError(f"non-JSON response (HTTP {resp.status_code}): "
                             f"{resp.text[:200]}") from exc


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_json(data: Any, compact: bool) -> None:
    if compact:
        print(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))


def main(argv) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    rest = argv[1:]
    compact = "--compact" in rest
    rest = [a for a in rest if a != "--compact"]

    try:
        client = IFindClient()
        if cmd == "token":
            tok = client.get_access_token(force=True)
            _print_json({"access_token": tok}, compact)
            return 0
        if cmd == "smoke":
            client.get_access_token(force=True)
            codes = "300033.SZ"
            ind = "latest"
            for i, a in enumerate(rest):
                if a == "--codes" and i + 1 < len(rest):
                    codes = rest[i + 1]
                if a == "--indicators" and i + 1 < len(rest):
                    ind = rest[i + 1]
            _print_json(client.post("real_time_quotation",
                                    {"codes": codes, "indicators": ind}), compact)
            return 0
        if cmd == "realtime":
            codes, ind = "300033.SZ", "latest"
            for i, a in enumerate(rest):
                if a == "--codes" and i + 1 < len(rest):
                    codes = rest[i + 1]
                if a == "--indicators" and i + 1 < len(rest):
                    ind = rest[i + 1]
            _print_json(client.post("real_time_quotation",
                                    {"codes": codes, "indicators": ind}), compact)
            return 0
        if cmd == "history":
            kw = _kv(rest, ["--codes", "--indicators", "--startdate", "--enddate"])
            for k in ("codes", "indicators", "startdate", "enddate"):
                if k not in kw:
                    raise IFindError(f"history requires --{k}")
            _print_json(client.post("cmd_history_quotation", {
                "codes": kw["codes"], "indicators": kw["indicators"],
                "startdate": kw["startdate"], "enddate": kw["enddate"],
                "functionpara": {"Fill": "Blank"},
            }), compact)
            return 0
        if cmd == "edb":
            kw = _kv(rest, ["--indicators", "--startdate", "--enddate"])
            for k in ("indicators", "startdate", "enddate"):
                if k not in kw:
                    raise IFindError(f"edb requires --{k}")
            _print_json(client.post("edb_service", {
                "indicators": kw["indicators"], "startdate": kw["startdate"],
                "enddate": kw["enddate"],
            }), compact)
            return 0
        if cmd == "post":
            if not rest:
                raise IFindError("usage: ifind.py post ENDPOINT --json '{...}'")
            endpoint = rest[0]
            payload = {}
            for i, a in enumerate(rest):
                if a == "--json" and i + 1 < len(rest):
                    payload = json.loads(rest[i + 1])
            _print_json(client.post(endpoint, payload), compact)
            return 0
        raise IFindError(f"unknown command: {cmd}. Run `ifind.py help`.")
    except IFindError as e:
        print(str(e), file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}", file=sys.stderr)
        return 2


def _kv(rest, flags) -> Dict[str, str]:
    """Parse --flag value pairs into a dict keyed by flag without dashes."""
    out = {}
    for i, a in enumerate(rest):
        if a in flags and i + 1 < len(rest):
            out[a.lstrip("-")] = rest[i + 1]
    return out


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
