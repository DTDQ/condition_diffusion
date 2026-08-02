#!/usr/bin/env python3
"""
Self-contained FRED (St. Louis Fed) US-macro connector + CLI for the quant-data skill.

FRED's CSV endpoint is free, no API key, no login:
    https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}

This covers US macro / global rates / credit spreads that Wind EDB and iFinD don't
surface cleanly:
  - US Treasuries (2Y/10Y), Fed funds, breakeven inflation
  - Credit spreads (HY OAS, IG OAS, AAA/BAA corporate yields)
  - Fed balance sheet

⚠️ Network note (from MidTermReport/ai_monitor): under some VPN setups Python
`requests` SSL-times-out against FRED while `curl` works. This connector tries
`requests` first, falls back to `curl` via subprocess. Any python3 works (no
WindPy/oracle deps).

Usage:
  python3 fred.py ping                       # connectivity: fetch one recent DGS10 point
  python3 fred.py series                     # list the curated FRED series catalog
  python3 fred.py fetch SERIES [--from YYYY-MM-DD] [--to ..] [--csv]
                                             # one or more FRED series ids (comma-joined)
  python3 fred.py bundle NAME [--from ..] [--to ..] [--csv]
                                             # named bundle: rates / credit / all

Output: padded table by default; --csv for CSV. One date column + one column per
series. FRED marks missing values as '.' — these become empty cells.
"""
import csv as _csv
import io
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from urllib.request import urlopen

_HERE = Path(__file__).resolve().parent

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# Curated catalog of US-macro / global-rate / credit-spread series used in
# MidTermReport/ai_monitor/crawlers/l1_macro.py. unit is FRED's native unit.
SERIES = {
    "DGS10":        {"name": "10Y美债收益率",       "unit": "%",      "bundle": "rates"},
    "DGS2":         {"name": "2Y美债收益率",        "unit": "%",      "bundle": "rates"},
    "FEDFUNDS":     {"name": "联邦基金利率",        "unit": "%",      "bundle": "rates"},
    "T10YIE":       {"name": "10Y盈亏平衡通胀",     "unit": "%",      "bundle": "rates"},
    "BAMLH0A0HYM2": {"name": "HY OAS 信用利差",     "unit": "%",      "bundle": "credit"},
    "BAMLC0A0CM":   {"name": "投资级公司债OAS",     "unit": "%",      "bundle": "credit"},
    "DAAA":         {"name": "AAA企业债收益率",     "unit": "%",      "bundle": "credit"},
    "DBAA":         {"name": "BAA企业债收益率",     "unit": "%",      "bundle": "credit"},
    "WALCL":        {"name": "美联储资产负债表",    "unit": "M USD",  "bundle": "fed_bs"},
}

BUNDLES = {
    "rates":   ["DGS10", "DGS2", "FEDFUNDS", "T10YIE"],
    "credit":  ["BAMLH0A0HYM2", "BAMLC0A0CM", "DAAA", "DBAA"],
    "fed_bs":  ["WALCL"],
    "all":     list(SERIES.keys()),
}


# --------------------------------------------------------------------------- #
# fetch (requests -> curl fallback)
# --------------------------------------------------------------------------- #
def _fetch_csv(series_id: str) -> str:
    """Return FRED CSV text for one series. Tries requests, falls back to curl.

    The curl fallback exists because some VPN environments SSL-time-out Python
    requests against FRED while curl works fine (per MidTermReport experience).
    """
    url = FRED_CSV.format(series_id=series_id)
    # 1) try requests
    try:
        import requests
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200 and resp.text:
            return resp.text
    except Exception:
        pass  # fall through to curl
    # 2) try urllib (stdlib, in case requests is missing)
    try:
        with urlopen(url, timeout=20) as r:
            text = r.read().decode("utf-8", errors="replace")
            if text:
                return text
    except Exception:
        pass
    # 3) curl fallback (most robust under VPN)
    for attempt in range(1, 4):
        try:
            result = subprocess.run(
                ["curl", "-sL", "--max-time", "20", url],
                capture_output=True, text=True, timeout=25,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout
        except subprocess.TimeoutExpired:
            pass
    raise SystemExit(f"[fred] failed to fetch {series_id} via requests/urllib/curl. "
                     "Check network/VPN.")


def _parse_csv(text: str, series_id: str):
    """Parse FRED CSV -> list of (date_str, value_or_None). FRED '.' = missing."""
    reader = _csv.reader(io.StringIO(text))
    header = next(reader, None)
    if not header or len(header) < 2:
        return []
    rows = []
    for row in reader:
        if len(row) < 2 or not row[0]:
            continue
        d, v = row[0], row[1]
        if v in (".", ""):
            rows.append((d, None))
        else:
            try:
                rows.append((d, float(v)))
            except ValueError:
                rows.append((d, None))
    return rows


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def _emit(cols, rows, as_csv):
    if as_csv:
        wr = _csv.writer(sys.stdout)
        wr.writerow(cols)
        wr.writerows(rows)
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


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_ping():
    """Connectivity check: fetch the most recent non-null DGS10 point."""
    rows = _parse_csv(_fetch_csv("DGS10"), "DGS10")
    # find last non-null
    for d, v in reversed(rows):
        if v is not None:
            print(f"[fred] OK. DGS10 {d} = {v}%  (FRED reachable)")
            return 0
    print("[fred] FRED reachable but DGS10 had no non-null values")
    return 0


def cmd_series():
    cols = ["series_id", "name", "unit", "bundle"]
    rows = [(sid, s["name"], s["unit"], s["bundle"]) for sid, s in SERIES.items()]
    _emit(cols, rows, False)
    return 0


def cmd_fetch(series_ids, d_from, d_to, as_csv):
    """Fetch one or more FRED series, align by date, emit one column per series."""
    sids = [s.strip() for s in series_ids.split(",") if s.strip()]
    # gather each series into {date: value}
    by_date = {}  # date -> {sid: value}
    labels = {}
    for sid in sids:
        meta = SERIES.get(sid, {"name": sid, "unit": ""})
        labels[sid] = meta["name"]
        for d, v in _parse_csv(_fetch_csv(sid), sid):
            if d_from and d < d_from:
                continue
            if d_to and d > d_to:
                continue
            by_date.setdefault(d, {})[sid] = v
    cols = ["date"] + [f"{sid} ({labels[sid]})" for sid in sids]
    rows = []
    for d in sorted(by_date):
        row = [d]
        for sid in sids:
            v = by_date[d].get(sid)
            row.append("" if v is None else v)
        rows.append(row)
    _emit(cols, rows, as_csv)
    return 0


def cmd_bundle(name, d_from, d_to, as_csv):
    name = name.lower()
    if name not in BUNDLES:
        raise SystemExit(f"[fred] unknown bundle: {name}. Available: " + ", ".join(BUNDLES))
    return cmd_fetch(",".join(BUNDLES[name]), d_from, d_to, as_csv)


def main(argv):
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        print("\nBundles: " + ", ".join(BUNDLES.keys()))
        return 0
    cmd = argv[0]
    rest = argv[1:]
    as_csv = "--csv" in rest
    rest = [a for a in rest if a != "--csv"]
    today = date.today().strftime("%Y-%m-%d")

    def _date_arg(rest, flag, default):
        for i, a in enumerate(rest):
            if a == flag and i + 1 < len(rest):
                return rest[i + 1]
        return default

    if cmd == "ping":
        return cmd_ping()
    if cmd == "series":
        return cmd_series()
    if cmd == "fetch":
        if not rest:
            raise SystemExit("usage: fred.py fetch SERIES [--from ..] [--to ..]")
        return cmd_fetch(rest[0], _date_arg(rest, "--from", "1990-01-01"),
                         _date_arg(rest, "--to", today), as_csv)
    if cmd == "bundle":
        if not rest:
            raise SystemExit("usage: fred.py bundle NAME [--from ..] [--to ..]")
        return cmd_bundle(rest[0], _date_arg(rest, "--from", "1990-01-01"),
                          _date_arg(rest, "--to", today), as_csv)
    raise SystemExit(f"[fred] unknown command: {cmd}. Run `fred.py help`.")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
