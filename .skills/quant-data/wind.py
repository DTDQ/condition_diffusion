#!/usr/bin/env python3
"""
Self-contained Wind desktop API (WindPy) connector + CLI for the quant-data skill.

This is the THIRD Wind access path (distinct from WDS Oracle in wds.py):
  - wds.py  -> Wind WDS Oracle DB at 222.66.94.9 (raw tables, /usr/bin/python3)
  - wind.py -> Wind desktop terminal via WindPy (EDB macro + wsd series, miniconda)

WindPy talks to the LOCAL Wind terminal (must be logged in), so it reaches data
the Oracle DB does not surface — most importantly the 经济数据库 (EDB) macro
indicators (CPI/PPI/M1/M2/social retail/central-bank rates/...) and the
multi-security multi-indicator time-series (w.wsd). iFinD's edb_service is gated
behind a monthly quota that is exhausted, so WindPy EDB is the working macro path.

⚠️ Interpreter: MUST run with **/Users/erleye/miniconda3/bin/python3** (3.13),
where WindPy is installed, and the Wind terminal must be running/logged in.
/usr/bin/python3 (3.8) has no WindPy and will fail the import.

Usage:
  /Users/erleye/miniconda3/bin/python3 wind.py ping
      Start WindPy, confirm connected to the terminal.
  /Users/erleye/miniconda3/bin/python3 wind.py edb CODES [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--csv]
      Macro EDB series. CODES = comma-joined Wind EDB indicators, e.g.
      M0001383,M0001385  (M1同比, M2同比). Defaults: from=2015-01-01, to=today.
  /Users/erleye/miniconda3/bin/python3 wind.py wsd CODES INDICATORS [--from ..] [--to ..] [--csv]
      Multi-security time-series. CODES = e.g. 000300.SH,600519.SH;
      INDICATORS = e.g. close,open,netprofit_ttm. Defaults: from=2015-01-01, to=today.
  /Users/erleye/miniconda3/bin/python3 wind.py macro NAME [--from ..] [--to ..] [--csv]
      Convenience: look up a NAMED macro bundle from data/macro_indicators.md catalog
      (e.g. `macro m1m2`, `macro cpi`, `macro cb_rates`, `macro retail`). Prints a
      labelled table instead of raw EDB codes.

Output: padded table by default; --csv for CSV. EDB returns one date column + one
column per code (labelled by code); wsd returns one date column + one column per
(code,indicator) pair.
"""
import csv as _csv
import sys
from datetime import date

# Named macro bundles — kept in sync with data/macro_indicators.md. Each value is
# a list of (code, label) pairs; labels become column headers in `macro` output.
MACRO_BUNDLES = {
    "m1m2": [
        ("M0001383", "M1同比(%)"),
        ("M0001385", "M2同比(%)"),
    ],
    "credit": [
        ("M0001383", "M1同比(%)"),
        ("M0001385", "M2同比(%)"),
        ("M5525763", "社融存量同比(%)"),
    ],
    "cpi": [
        ("M0001227", "PPI当月同比(%)"),   # PPI lives next to CPI in practice
        ("M0001212", "CPI当月同比(%)"),
    ],
    "ppi": [
        ("M0001227", "PPI当月同比(%)"),
    ],
    "retail": [
        ("M0001428", "社零当月同比(%)"),
        ("M0001427", "社零当月值(亿元)"),
        ("M0001426", "社零累计同比(%)"),
    ],
    "cb_rates": [
        ("M0000162", "FED联邦基金目标利率上限(%)"),
        ("M0000164", "BOJ政策利率(%)"),
        ("M0000166", "ECB主要再融资利率MRO(%)"),
        ("G0006339", "BOE银行利率(%)"),
        ("G1600268", "RBA现金利率(%)"),
        ("M0329545", "PBOC MLF 1Y(%)"),
    ],
    "corp_loan": [
        ("M0043417", "企业中长贷余额(亿)"),
        ("M0057877", "企业中长贷当月新增(亿)"),
    ],
}


def _get_wind():
    """Import WindPy (miniconda) and start it. Raises SystemExit with guidance."""
    try:
        from WindPy import w
    except ImportError:
        raise SystemExit(
            "[wind] WindPy not found. It lives in miniconda python3.13 — run with:\n"
            "  /Users/erleye/miniconda3/bin/python3 " + str(_here() / "wind.py") + " ...\n"
            "Also ensure the Wind terminal is running and logged in.")
    r = w.start(waitTime=30)
    if r.ErrorCode != 0 and not w.isconnected():
        raise SystemExit(f"[wind] WindPy start failed (ErrorCode={r.ErrorCode}). "
                         "Is the Wind terminal running and logged in?")
    return w


def _here():
    from pathlib import Path
    return Path(__file__).resolve().parent


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


def _fmt_date(d):
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    return str(d)


def cmd_ping():
    w = _get_wind()
    print(f"[wind] connected: {w.isconnected()}")
    w.stop()
    return 0


def cmd_edb(codes, d_from, d_to, as_csv):
    w = _get_wind()
    try:
        code_list = [c.strip() for c in codes.split(",") if c.strip()]
        d = w.edb(codes, d_from, d_to, "Fill=Previous")
        if d.ErrorCode != 0:
            raise SystemExit(f"[wind] w.edb failed ErrorCode={d.ErrorCode}: {d.Data}")
        cols = ["date"] + [c for c in (d.Codes if hasattr(d, "Codes") and d.Codes else code_list)]
        rows = []
        for i, dt in enumerate(d.Times):
            row = [_fmt_date(dt)]
            for j in range(len(code_list)):
                v = d.Data[j][i] if j < len(d.Data) else None
                row.append("" if v is None else v)
            rows.append(row)
        _emit(cols, rows, as_csv)
    finally:
        w.stop()
    return 0


def cmd_wsd(codes, indicators, d_from, d_to, as_csv):
    w = _get_wind()
    try:
        code_list = [c.strip() for c in codes.split(",") if c.strip()]
        ind_list = [i.strip() for i in indicators.split(",") if i.strip()]
        d = w.wsd(codes, indicators, d_from, d_to, "Fill=Previous")
        if d.ErrorCode != 0:
            raise SystemExit(f"[wind] w.wsd failed ErrorCode={d.ErrorCode}: {d.Data}")
        # Build column headers: date, then CODE.INDICATOR for each combination.
        # d.Codes/Indicators may be populated; fall back to the requested lists.
        ret_codes = list(d.Codes) if hasattr(d, "Codes") and d.Codes else code_list
        ret_inds = list(getattr(d, "Indicators", None) or ind_list)
        cols = ["date"]
        # Data is laid out as one column per (code, indicator), code-major.
        for ci, _code in enumerate(ret_codes):
            for ii, ind in enumerate(ret_inds):
                idx = ci * len(ret_inds) + ii
                cols.append(f"{ret_codes[ci]}.{ind}" if idx < (len(ret_codes) * len(ret_inds)) else f"col{idx}")
        rows = []
        for i, dt in enumerate(d.Times):
            row = [_fmt_date(dt)]
            for j in range(len(cols) - 1):
                v = d.Data[j][i] if j < len(d.Data) else None
                row.append("" if v is None else v)
            rows.append(row)
        _emit(cols, rows, as_csv)
    finally:
        w.stop()
    return 0


def cmd_macro(name, d_from, d_to, as_csv):
    name = name.lower()
    if name not in MACRO_BUNDLES:
        raise SystemExit(f"[wind] unknown macro bundle: {name}. Available: "
                         + ", ".join(MACRO_BUNDLES.keys()))
    pairs = MACRO_BUNDLES[name]
    codes = ",".join(c for c, _ in pairs)
    w = _get_wind()
    try:
        d = w.edb(codes, d_from, d_to, "Fill=Previous")
        if d.ErrorCode != 0:
            raise SystemExit(f"[wind] w.edb failed ErrorCode={d.ErrorCode}: {d.Data}")
        cols = ["date"] + [lbl for _, lbl in pairs]
        rows = []
        for i, dt in enumerate(d.Times):
            row = [_fmt_date(dt)]
            for j in range(len(pairs)):
                v = d.Data[j][i] if j < len(d.Data) else None
                row.append("" if v is None else v)
            rows.append(row)
        _emit(cols, rows, as_csv)
    finally:
        w.stop()
    return 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        print("\nNamed macro bundles: " + ", ".join(MACRO_BUNDLES.keys()))
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
    if cmd == "edb":
        if not rest:
            raise SystemExit("usage: wind.py edb CODES [--from ..] [--to ..]")
        codes = rest[0]
        return cmd_edb(codes, _date_arg(rest, "--from", "2015-01-01"), _date_arg(rest, "--to", today), as_csv)
    if cmd == "wsd":
        if len(rest) < 2:
            raise SystemExit("usage: wind.py wsd CODES INDICATORS [--from ..] [--to ..]")
        return cmd_wsd(rest[0], rest[1], _date_arg(rest, "--from", "2015-01-01"), _date_arg(rest, "--to", today), as_csv)
    if cmd == "macro":
        if not rest:
            raise SystemExit("usage: wind.py macro NAME [--from ..] [--to ..]")
        return cmd_macro(rest[0], _date_arg(rest, "--from", "2015-01-01"), _date_arg(rest, "--to", today), as_csv)
    raise SystemExit(f"[wind] unknown command: {cmd}. Run `wind.py help`.")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
