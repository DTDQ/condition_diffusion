#!/usr/bin/env python3
"""
Self-contained Factor DB (MySQL) connector + CLI for the quant-data skill.

Connects to the team-computed quant database at 222.66.94.9:33060 (user jrgc2),
which hosts four quant schemas:
  - `market`          : 中国A股基础数据库 (26 tables: trading_calendar, stock_desc,
                        stock_market, stock_balance, stock_income, stock_cashflow,
                        stock_status, stock_ttm, stock_dividend, stock_money_flow,
                        fama_french, monthly/weekly_stock_market, ...)
  - `factors`         : 中国A股因子数据库 (17 tables: factor_meta, barra_descriptors,
                        barra_factors, barra_factor_return, barra_risk_forcast,
                        specific_volatility, alpha_factors, alpha_*_descriptors,
                        factor_corr, factor_indicators_*, factor_performance_*)
  - `mutualfunddata`  : 公募基金数据库 (alphaload/return, industryload/return,
                        styleload/return, totalreturn)
  - `wyckoff_tech`    : 威科夫技术指标库 (wso, wss)

Note: the factor schema is named `factors` (plural). The default DB for `sql`/
`tables` is `market`; pass `--db factors` (or mutualfunddata / wyckoff_tech).

Reads credentials from (first found):
  1. ~/.zcode/skills/quant-data/factor_credentials.env
  2. environment variables FACTOR_HOST / FACTOR_PORT / FACTOR_USER / FACTOR_PASSWORD

Usage:
  python3 factor.py ping                 # test connection + list databases
  python3 factor.py dbs                  # list databases
  python3 factor.py tables [DB]          # list tables in a database (default market)
  python3 factor.py cols DB.TABLE        # DESCRIBE a table
  python3 factor.py sql "SELECT ..." [...] [--db factors|market|mutualfunddata|wyckoff_tech] [--limit N] [--csv]
                                         # read-only SQL (SELECT/SHOW/DESC/EXPLAIN only)
  python3 factor.py schema [DB|TABLE]    # print curated schema from data/factor_schema.json

Output is UTF-8 text; --csv emits CSV, otherwise a padded table.
"""
import os
import re
import sys
import json
import csv as _csv
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CRED_FILE = _HERE / "factor_credentials.env"
_SCHEMA_FILE = _HERE / "data" / "factor_schema.json"

# SQL statements allowed by the read-only guard. Anything else (INSERT/UPDATE/
# DELETE/DDL) is rejected before it reaches the server.
_READONLY_RE = re.compile(
    r"^\s*(SELECT|SHOW|DESCRIBE|DESC|EXPLAIN|WITH)\b",
    re.IGNORECASE,
)


def _load_creds() -> dict:
    cfg = {}
    if _CRED_FILE.exists():
        for line in _CRED_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            cfg[k.strip()] = v.strip()
    for k in ("FACTOR_HOST", "FACTOR_PORT", "FACTOR_USER", "FACTOR_PASSWORD"):
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    return cfg


def get_connection(database: str = None):
    """Return a pymysql connection to the Factor DB.

    `database` selects the initial schema (e.g. 'factors' or 'market'); None
    connects without a default schema (use for `dbs`).
    """
    try:
        import pymysql
    except ImportError:
        raise SystemExit(
            "[factor] pymysql is not installed. Install with:\n"
            "  /usr/bin/python3 -m pip install pymysql"
        )
    c = _load_creds()
    missing = [k for k in ("FACTOR_HOST", "FACTOR_PORT", "FACTOR_USER", "FACTOR_PASSWORD")
               if not c.get(k)]
    if missing:
        raise SystemExit(f"[factor] missing credentials: {missing}. "
                         f"Set them in {_CRED_FILE} or env vars.")
    return pymysql.connect(
        host=c["FACTOR_HOST"],
        port=int(c["FACTOR_PORT"]),
        user=c["FACTOR_USER"],
        password=c["FACTOR_PASSWORD"],
        database=database,
        charset="utf8mb4",
        connect_timeout=10,
        read_timeout=60,
        cursorclass=pymysql.cursors.Cursor,
    )


# --------------------------------------------------------------------------- #
# output helpers (shared shape with wds.py)
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


def _run_sql(sql: str, db: str, binds, limit, as_csv):
    if not _READONLY_RE.match(sql):
        raise SystemExit("[factor] rejected: only SELECT/SHOW/DESCRIBE/EXPLAIN/WITH "
                         "are allowed (read-only).")
    conn = get_connection(db)
    try:
        cur = conn.cursor()
        # pymysql uses %s placeholders; pass binds as a tuple. None/[] -> no args.
        cur.execute(sql, tuple(binds) if binds else None)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(limit) if limit else cur.fetchall()
        cur.close()
    finally:
        conn.close()
    _emit(cols, rows, as_csv)


# --------------------------------------------------------------------------- #
# schema (offline curated schema from factor_schema.json)
# --------------------------------------------------------------------------- #
def _load_schema() -> dict:
    if not _SCHEMA_FILE.exists():
        raise SystemExit(f"[factor] schema not found: {_SCHEMA_FILE}. "
                         "Run tools/build_factor_schema.py first.")
    return json.loads(_SCHEMA_FILE.read_text(encoding="utf-8"))


def cmd_schema(args):
    schema = _load_schema()
    if not args:
        for db_name, db in schema.items():
            print(f"{db_name}  —  {db['title']}  ({len(db['tables'])} tables)")
            for tname, t in sorted(db["tables"].items()):
                print(f"    {tname:30} {t.get('comment') or ''}  ({len(t['fields'])} fields)")
        return
    target = args[0]
    # is it a database?
    if target in schema:
        db = schema[target]
        print(f"# {target} — {db['title']}")
        for tname, t in sorted(db["tables"].items()):
            print(f"\n## {tname}  ({len(t['fields'])} fields)  {t.get('comment') or ''}")
            rows = [(f["name"], f["type"], f["cn"]) for f in t["fields"]]
            _emit(["field", "type", "cn"], rows, False)
        return
    # else: a table name
    for db_name, db in schema.items():
        if target in db["tables"]:
            t = db["tables"][target]
            print(f"# {target}  ({db_name}, {len(t['fields'])} fields)  {t.get('comment') or ''}")
            rows = [(f["name"], f["type"], f["cn"]) for f in t["fields"]]
            _emit(["field", "type", "cn"], rows, False)
            return
    raise SystemExit(f"[factor] schema target not found: {target}. "
                     "Run `schema` to list dbs/tables.")


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
        try:
            cur = conn.cursor()
            cur.execute("SHOW DATABASES")
            dbs = [r[0] for r in cur.fetchall()]
            cur.close()
        finally:
            conn.close()
        print(f"[factor] connected. MySQL server: {conn.get_server_info()}")
        print("databases:")
        for db in dbs:
            print(f"  - {db}")
        return 0

    if cmd == "dbs":
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SHOW DATABASES")
            rows = cur.fetchall()
            cur.close()
        finally:
            conn.close()
        _emit(["database"], rows, as_csv)
        return 0

    if cmd == "tables":
        db = "market"
        if rest:
            db = rest[0]
        _run_sql("SHOW TABLES", db, [], 2000, as_csv)
        return 0

    if cmd == "cols":
        if not rest:
            raise SystemExit("usage: factor.py cols DB.TABLE")
        db, _, table = rest[0].partition(".")
        if not table:
            raise SystemExit("usage: factor.py cols DB.TABLE  (e.g. market.stock_market)")
        _run_sql(f"DESCRIBE `{db}`.`{table}`", db, [], 0, as_csv)
        return 0

    if cmd == "sql":
        if not rest:
            raise SystemExit('usage: factor.py sql "SELECT ..." [--db factors|market|...] [--limit N]')
        sql = rest[0]
        db = "market"
        limit = 200
        binds = []
        i = 1
        while i < len(rest):
            if rest[i] == "--db" and i + 1 < len(rest):
                db = rest[i + 1]; i += 2; continue
            if rest[i] == "--limit" and i + 1 < len(rest):
                limit = int(rest[i + 1]); i += 2; continue
            binds.append(rest[i]); i += 1   # positional bind args
        _run_sql(sql, db, binds, limit, as_csv)
        return 0

    if cmd == "schema":
        cmd_schema(rest)
        return 0

    raise SystemExit(f"[factor] unknown command: {cmd}. Run `factor.py help`.")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
