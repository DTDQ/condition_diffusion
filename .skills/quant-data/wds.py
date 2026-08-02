#!/usr/bin/env python3
"""
Self-contained Wind WDS (Oracle) connector + CLI for the quant-data skill.

Reads credentials from (first found):
  1. ~/.zcode/skills/quant-data/wds_credentials.env
  2. environment variables WIND_HOST / WIND_PORT / WIND_SERVICE / WIND_USER /
     WIND_PASSWORD / ORACLE_CLIENT_DIR

The data dictionary (`dict` subcommands) is read from data/wds_dictionary.json
and needs NO database connection — works with any interpreter.

Usage:
  python3 wds.py ping
      Test the connection; print Oracle server version.
  python3 wds.py sql "SELECT ... :1 ..." [bind1 bind2 ...] [--limit N] [--csv]
      Run a read-only query with positional binds (:1, :2, ...). Default 200 rows.
  python3 wds.py tables [LIKE]        # list winddb tables, optional name filter
  python3 wds.py cols  TABLE          # list columns of winddb.TABLE
  python3 wds.py index-eod CODE [--from YYYYMMDD] [--to YYYYMMDD] [--csv]
      OHLCV for an index. Auto-routes CITIC industry codes (CI*) to
      aindexindustrieseodcitics, broad-market codes to aindexeodprices.
  python3 wds.py dict categories
  python3 wds.py dict list [CATEGORY]
  python3 wds.py dict table NAME
  python3 wds.py dict search 关键词
      Query the cleaned WDS data dictionary (offline, no DB needed).

Output is UTF-8 text; --csv emits CSV, otherwise a padded table.
"""
import os
import sys
import json
import csv as _csv
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CRED_FILE = _HERE / "wds_credentials.env"
_DICT_FILE = _HERE / "data" / "wds_dictionary.json"


def _load_creds() -> dict:
    cfg = {}
    if _CRED_FILE.exists():
        for line in _CRED_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            cfg[k.strip()] = v.strip()
    # env overrides file
    for k in ("WIND_HOST", "WIND_PORT", "WIND_SERVICE", "WIND_USER",
              "WIND_PASSWORD", "ORACLE_CLIENT_DIR"):
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    return cfg


def get_connection():
    """Return an oracledb connection to Wind WDS (thick mode if client dir set)."""
    import oracledb
    c = _load_creds()
    missing = [k for k in ("WIND_HOST", "WIND_PORT", "WIND_SERVICE", "WIND_USER", "WIND_PASSWORD")
               if not c.get(k)]
    if missing:
        raise SystemExit(f"[wds] missing credentials: {missing}. "
                         f"Set them in {_CRED_FILE} or env vars.")
    client_dir = c.get("ORACLE_CLIENT_DIR", "")
    if client_dir:
        try:
            oracledb.init_oracle_client(lib_dir=client_dir)
        except Exception:
            pass  # already initialised, or fall back to thin mode
    dsn = f"{c['WIND_HOST']}:{c['WIND_PORT']}/{c['WIND_SERVICE']}"
    return oracledb.connect(user=c["WIND_USER"], password=c["WIND_PASSWORD"], dsn=dsn)


def wind_code(symbol: str) -> str:
    """Normalise to Wind code: 600519.SS -> 600519.SH; CI/others pass through."""
    s = symbol.strip()
    if "." not in s:
        return s.upper()
    code, suf = s.rsplit(".", 1)
    suf = suf.upper()
    return f"{code}.{'SH' if suf == 'SS' else suf}"


# --------------------------------------------------------------------------- #
# output helpers
# --------------------------------------------------------------------------- #
def _emit(cols, rows, as_csv: bool):
    if as_csv:
        w = _csv.writer(sys.stdout)
        w.writerow(cols)
        w.writerows(rows)
        return
    widths = [len(str(c)) for c in cols]
    for r in rows:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], len(str(v)))
    print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cols)))
    print("  ".join("-" * widths[i] for i in range(len(cols))))
    for r in rows:
        print("  ".join(str(v).ljust(widths[i]) for i, v in enumerate(r)))
    print(f"[{len(rows)} rows]")


def _run_sql(sql: str, binds, limit, as_csv):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(sql, binds or [])
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchmany(limit) if limit else cur.fetchall()
    cur.close()
    conn.close()
    _emit(cols, rows, as_csv)


# --------------------------------------------------------------------------- #
# dict (offline dictionary query)
# --------------------------------------------------------------------------- #
def _load_dict() -> dict:
    if not _DICT_FILE.exists():
        raise SystemExit(f"[wds] dictionary not found: {_DICT_FILE}. "
                         "Run tools/clean_wds_dict.py first.")
    return json.loads(_DICT_FILE.read_text(encoding="utf-8"))


def _find_table(d: dict, name: str):
    """Return (category, table_obj) for a table name (case-insensitive)."""
    target = name.upper()
    for cat, tables in d.items():
        for tname, t in tables.items():
            if tname.upper() == target:
                return cat, t
    return None, None


def cmd_dict(args):
    if not args:
        raise SystemExit("usage: wds.py dict {categories|list [CAT]|table NAME|search KW}")
    sub = args[0]
    d = _load_dict()
    if sub == "categories":
        for cat, tables in d.items():
            print(f"{len(tables):>3}  {cat}")
        print(f"[{len(d)} categories]")
        return
    if sub == "list":
        cat_arg = args[1] if len(args) > 1 else None
        rows = []
        if cat_arg:
            cat_arg_norm = cat_arg
            if cat_arg_norm not in d:
                # fuzzy: substring match
                matches = [c for c in d if cat_arg in c]
                if len(matches) != 1:
                    raise SystemExit(f"[wds] category not found: {cat_arg}. "
                                     f"Run `dict categories` to list. "
                                     f"{'Matches: ' + str(matches) if matches else ''}")
                cat_arg_norm = matches[0]
            for tname, t in sorted(d[cat_arg_norm].items()):
                rows.append((tname, len(t["fields"]),
                             ", ".join(f["name"] for f in t["fields"][:4])))
            _emit(["table", "fields", "key_fields"], rows, False)
            return
        # all categories
        for cat, tables in d.items():
            for tname, t in sorted(tables.items()):
                rows.append((cat, tname, len(t["fields"])))
        _emit(["category", "table", "fields"], rows, False)
        return
    if sub == "table":
        if len(args) < 2:
            raise SystemExit("usage: wds.py dict table NAME")
        cat, t = _find_table(d, args[1])
        if t is None:
            raise SystemExit(f"[wds] table not found: {args[1]}. Run `dict list`.")
        print(f"# {args[1].upper()}  ({cat}, {len(t['fields'])} fields)")
        print(t.get("description", ""))
        rows = [(f["name"], f["type"], f["cn"], f["fill_rate"], f["desc"])
                for f in t["fields"]]
        _emit(["field", "type", "cn", "fill_rate", "desc"], rows, False)
        return
    if sub == "search":
        if len(args) < 2:
            raise SystemExit("usage: wds.py dict search KEYWORD")
        kw = args[1].lower()
        hits = []
        for cat, tables in d.items():
            for tname, t in tables.items():
                for f in t["fields"]:
                    hay = " ".join([f["name"], f["cn"], f["desc"]]).lower()
                    if kw in hay:
                        hits.append((tname, f["name"], f["cn"], f["type"]))
        _emit(["table", "field", "cn", "type"], hits, False)
        return
    raise SystemExit(f"[wds] unknown dict subcommand: {sub}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv):
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    rest = argv[1:]
    as_csv = "--csv" in rest
    rest = [a for a in rest if a != "--csv"]

    if cmd == "ping":
        conn = get_connection()
        print(f"[wds] connected: Oracle {conn.version}")
        conn.close()
        return 0

    if cmd == "tables":
        like = rest[0] if rest else None
        sql = ("SELECT table_name FROM all_tables WHERE owner='WINDDB'"
               + (" AND table_name LIKE :1" if like else "")
               + " ORDER BY table_name")
        _run_sql(sql, [like.upper()] if like else [], 1000, as_csv)
        return 0

    if cmd == "cols":
        if not rest:
            raise SystemExit("usage: wds.py cols TABLE")
        sql = ("SELECT column_name, data_type, data_length FROM all_tab_columns "
               "WHERE owner='WINDDB' AND table_name=:1 ORDER BY column_id")
        _run_sql(sql, [rest[0].upper()], 0, as_csv)
        return 0

    if cmd == "index-eod":
        if not rest:
            raise SystemExit("usage: wds.py index-eod CODE [--from YYYYMMDD] [--to YYYYMMDD]")
        code = wind_code(rest[0])
        d_from = d_to = None
        for i, a in enumerate(rest):
            if a == "--from" and i + 1 < len(rest):
                d_from = rest[i + 1]
            if a == "--to" and i + 1 < len(rest):
                d_to = rest[i + 1]
        table = ("winddb.aindexindustrieseodcitics" if code.upper().startswith("CI")
                 else "winddb.aindexeodprices")
        sql = (f"SELECT TRADE_DT, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE, S_DQ_VOLUME "
               f"FROM {table} WHERE S_INFO_WINDCODE = :1 AND S_DQ_CLOSE IS NOT NULL")
        binds = [code]
        if d_from:
            sql += f" AND TRADE_DT >= :{len(binds)+1}"; binds.append(d_from)
        if d_to:
            sql += f" AND TRADE_DT <= :{len(binds)+1}"; binds.append(d_to)
        sql += " ORDER BY TRADE_DT"
        _run_sql(sql, binds, 0, as_csv)
        return 0

    if cmd == "sql":
        if not rest:
            raise SystemExit('usage: wds.py sql "SELECT ..." [binds...] [--limit N]')
        sql = rest[0]
        limit = 200
        binds = []
        i = 1
        while i < len(rest):
            if rest[i] == "--limit" and i + 1 < len(rest):
                limit = int(rest[i + 1]); i += 2; continue
            binds.append(rest[i]); i += 1
        _run_sql(sql, binds, limit, as_csv)
        return 0

    if cmd == "dict":
        cmd_dict(rest)
        return 0

    raise SystemExit(f"[wds] unknown command: {cmd}. Run `wds.py help`.")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
