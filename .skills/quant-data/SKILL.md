---
name: quant-data
description: Unified quant-research data-access skill for five sources — (1) Wind WDS (万得) Oracle raw data via winddb schema, (2) the team's self-computed Factor DB (MySQL) with `factors` (Barra/Alpha factors) and `market` (A-share basics) schemas, (3) Tonghuashun iFinD (同花顺) HTTP API with automatic token refresh, (4) Wind desktop API (WindPy) for China macro EDB indicators (CPI/PPI/M1/M2/社零/央行利率) and multi-security wsd series, and (5) FRED (St. Louis Fed) for US macro / rates / credit spreads (free CSV, no key). Use whenever the user wants quant data, A-share prices, financial statements, Barra/alpha factors, index constituents, CITIC industry indices, macro/宏观经济 data (China or US), realtime quotes, research reports, smart stock-picking, or anything from Wind/万得/wds, the factor/因子 database, or iFind/同花顺. Provides offline data-dictionary queries too.
---

# Quant Data (WDS + Factor DB + iFinD + WindPy + FRED)

A system-wide, self-contained connector to the five data sources used in
quant research. Credentials live in `*.env` files next to this skill
(chmod 600); WindPy needs no credential file but requires the Wind terminal
logged in; FRED needs no credentials at all. No project setup needed.

```
~/.zcode/skills/quant-data/
├── SKILL.md
├── wds.py / wds_credentials.env        # Wind WDS (Oracle, schema winddb)
├── factor.py / factor_credentials.env  # Factor DB (MySQL: factors + market)
├── ifind.py / ifind.env                # iFinD HTTP API (auto token refresh)
├── wind.py                             # Wind desktop API (WindPy: China macro EDB + wsd)
├── fred.py                             # FRED US macro (CSV, no key, requests→curl fallback)
├── data/
│   ├── wds_dictionary.json / .md       # cleaned WDS dict (151 tables)
│   ├── factor_schema.json / .md        # live-introspected schema (52 tables)
│   ├── ifind_query_manual.md           # iFinD endpoint reference
│   └── macro_indicators.md             # Wind EDB + FRED macro catalog
└── tools/                              # reproducible dict/schema builders
```

## Which source to use (routing)

| Need | Source | Examples |
| --- | --- | --- |
| Raw Wind data — A-share OHLCV, indices, financial statements, valuation, CITIC industry, calendar | **WDS** (`wds.py`) | `winddb.ashareeodprices`, `aindexeodprices`, `aindexindustrieseodcitics`, `asharebalancesheet` |
| Team-computed Barra/Alpha factors, A-share basics the team pre-computed (clean stock_market, stock_ttm, barra_factors, alpha_factors) | **Factor DB** (`factor.py`) | `market.stock_market`, `market.stock_ttm`, `factors.barra_factors`, `factors.alpha_factors` |
| **China macro/中国宏观 (CPI/PPI/M1/M2/社零/社融/央行利率/企业信贷)**, multi-security wsd series (TTM netprofit, any indicator across many codes) | **WindPy** (`wind.py`) | `macro m1m2`, `macro cb_rates`, `edb M0001383`, `wsd CI005021.WI netprofit_ttm` |
| **US macro/美国宏观 (美债/联邦基金利率/信用利差/美联储资产负债表)** | **FRED** (`fred.py`) | `bundle rates`, `bundle credit`, `fetch DGS10,DGS2` |
| Realtime quotes, high-frequency, research reports, index constituents, smart stock-picking (问财), anything Wind WDS lacks | **iFinD** (`ifind.py`) | `real_time_quotation`, `cmd_history_quotation`, `report_query`, `data_pool`, `smart_stock_picking` |

Rule of thumb: Wind raw tables → WDS; team factor computations → factor DB;
**中国宏观/EDB 或多证券多指标 wsd → WindPy** (`wind.py`); **美国宏观/利率/信用利差 → FRED**
(`fred.py`); realtime/reports/智能选股 → iFinD. Note: iFinD's `edb_service` is
quota-limited (currently exhausted), so macro data goes through WindPy (China) +
FRED (US). When unsure, ask the user.
| Realtime quotes, high-frequency, macro EDB, research reports, index constituents, smart stock-picking (问财), anything Wind WDS lacks | **iFinD** (`ifind.py`) | `real_time_quotation`, `cmd_history_quotation`, `edb_service`, `report_query`, `data_pool`, `smart_stock_picking` |

Rule of thumb: Wind raw data → WDS; team factor computations → factor DB;
macro/realtime/reports/智能选股 → iFinD. When unsure, ask the user.

---

## 1. WDS — Wind Oracle (schema `winddb`)

```bash
/usr/bin/python3 ~/.zcode/skills/quant-data/wds.py ping
/usr/bin/python3 ~/.zcode/skills/quant-data/wds.py tables [LIKE]
/usr/bin/python3 ~/.zcode/skills/quant-data/wds.py cols  TABLE
/usr/bin/python3 ~/.zcode/skills/quant-data/wds.py index-eod CODE [--from YYYYMMDD] [--to YYYYMMDD] [--csv]
/usr/bin/python3 ~/.zcode/skills/quant-data/wds.py sql "SELECT ... :1 ..." [binds...] [--limit N] [--csv]
# offline data dictionary (no DB connection, any interpreter):
/usr/bin/python3 ~/.zcode/skills/quant-data/wds.py dict categories
/usr/bin/python3 ~/.zcode/skills/quant-data/wds.py dict list [CATEGORY]
/usr/bin/python3 ~/.zcode/skills/quant-data/wds.py dict table NAME
/usr/bin/python3 ~/.zcode/skills/quant-data/wds.py dict search 关键词
```

⚠️ **Interpreter**: WDS commands that touch the DB (`ping/tables/cols/index-eod/sql`)
must run with **`/usr/bin/python3`** (Python 3.8, thick mode via instantclient) —
miniconda's 3.13 uses oracledb thin mode which can't connect to the old 11.2 Oracle
(DPY-3010). The `dict` subcommands are pure JSON and work with any python3.

Key tables in `winddb`:
- A-share daily OHLCV (back-adjusted): `ashareeodprices`
  (`S_DQ_ADJOPEN/ADJHIGH/ADJLOW/ADJCLOSE`, `S_DQ_VOLUME`; key `S_INFO_WINDCODE`,
  `TRADE_DT` as `YYYYMMDD`).
- Broad-market index OHLCV (raw): `aindexeodprices`.
- **CITIC (中信) industry index OHLCV: `aindexindustrieseodcitics`** (NOT aindexeodprices).
  `index-eod` auto-routes `CI*` codes here. CITIC L1 = `CI005001.WI`..`CI005030.WI`.
- `TRADE_DT` is a string `YYYYMMDD`. Codes use `.SH`/`.SZ`/`.BJ` (Yahoo `.SS` → `.SH`).

The cleaned dictionary (`data/wds_dictionary.json`) covers 16 categories / 151 tables /
1610 fields. Browse `data/wds_dictionary_index.md` or query with `dict`.

Notes: read-only (never DML/DDL). The Wind server occasionally drops connections
(ORA-12537 / DPY-4011) — retry `ping` then the query.

---

## 2. Factor DB — team MySQL (schemas `factor` + `market`)

```bash
/usr/bin/python3 ~/.zcode/skills/quant-data/factor.py ping
/usr/bin/python3 ~/.zcode/skills/quant-data/factor.py dbs
/usr/bin/python3 ~/.zcode/skills/quant-data/factor.py tables [DB]            # default market
/usr/bin/python3 ~/.zcode/skills/quant-data/factor.py cols DB.TABLE
/usr/bin/python3 ~/.zcode/skills/quant-data/factor.py sql "SELECT ..." [--db factors|market|mutualfunddata|wyckoff_tech] [--limit N] [--csv]
/usr/bin/python3 ~/.zcode/skills/quant-data/factor.py schema [DB|TABLE]      # curated schema (live-introspected)
```

Requires `pymysql` — if missing, install: `/usr/bin/python3 -m pip install pymysql`.
The factor schema is named **`factors`** (plural). Default DB for `sql`/`tables`
is `market`; pass `--db factors` for factors. The curated schema in
`data/factor_schema.json` is rebuilt from the **live** database (authoritative) +
PDF Chinese notes — see `factor.py schema`.

Four quant schemas (52 tables total, 2026-07 introspection):
- **`market`** — 中国A股基础数据库 (26 tables): `trading_calendar`, `stock_desc`,
  `index_desc`, `stock_market` (daily OHLCV + adj + 涨跌停 + vwap_30min),
  `monthly_stock_market`, `weekly_stock_market`, `stock_equity`, `stock_balance`,
  `stock_income`, `stock_cashflow`, `stock_status`, `stock_report_status`,
  `stock_ttm`, `stock_rolling_consensus`, `stock_dividend`, `stock_money_flow`,
  `stock_management`, `stock_rating_consus`, `fama_french`, ...
- **`factors`** — 中国A股因子数据库 (17 tables): `factor_meta`, `barra_descriptors`
  (style descriptors: beta, lncap, btop, …), `barra_factors` (10 big styles +
  CITIC industry dummies), `barra_factor_return`, `barra_risk_forcast` (JSON
  daily/weekly/monthly), `specific_volatility`, `alpha_factors` (composite alphas),
  `alpha_financial_descriptors`, `alpha_std_descriptors`, `alpha_trading_descriptors`,
  `factor_corr`, `factor_indicators_{monthly,weekly,range_*}`, `factor_performance_{monthly,weekly}`.
- **`mutualfunddata`** — 公募基金数据库 (7 tables): `alphaload`/`alphareturn`,
  `industryload`/`industryreturn`, `styleload`/`stylereturn`, `totalreturn`.
- **`wyckoff_tech`** — 威科夫技术指标库 (2 tables): `wso`, `wss`.

Common keys: `date` (DATE), `stock_id` (VARCHAR(30), Wind code without exchange
suffix — verify with `cols`). Full schema in `data/factor_schema.json` / `.md`,
or `factor.py schema`.

⚠️ Read-only guard: `sql` only accepts `SELECT/SHOW/DESCRIBE/EXPLAIN/WITH`. No
INSERT/UPDATE/DELETE/DDL.

---

## 3. WindPy — Wind desktop API (macro EDB + multi-security wsd)

```bash
PY=/Users/erleye/miniconda3/bin/python3
SKILL=~/.zcode/skills/quant-data
$PY $SKILL/wind.py ping                                          # start WindPy, confirm connected
$PY $SKILL/wind.py macro NAME [--from YYYY-MM-DD] [--to ..] [--csv]   # named macro bundle
$PY $SKILL/wind.py edb CODES [--from ..] [--to ..] [--csv]        # arbitrary EDB indicators
$PY $SKILL/wind.py wsd CODES INDICATORS [--from ..] [--to ..] [--csv]  # multi-security series
```

⚠️ **Interpreter**: MUST run with **`/Users/erleye/miniconda3/bin/python3`** (Python 3.13,
where WindPy is installed) and the **Wind terminal must be running and logged in**.
`/usr/bin/python3` (3.8) has no WindPy. This is a DIFFERENT access path from WDS
Oracle (`wds.py` uses `/usr/bin/python3`); don't mix the interpreters.

This is the working macro path: iFinD's `edb_service` is gated behind a monthly
quota that is currently exhausted (errorcode -4318), so 宏观经济数据 (CPI/PPI/
M1/M2/社零/社融/央行利率/企业信贷) goes through WindPy `w.edb`.

Named macro bundles (`macro NAME`, full catalog in `data/macro_indicators.md`):
- `m1m2` M1/M2 同比 · `credit` M1/M2/社融 · `cpi` CPI/PPI · `ppi` PPI
- `retail` 社零当月同比/值/累计 · `cb_rates` FED/BOJ/ECB/BOE/RBA/PBOC-MLF
- `corp_loan` 企业中长贷余额/当月新增

EDB codes: `M` prefix = China macro, `G` prefix = global macro. Examples:
`M0001383` M1同比, `M0001385` M2同比, `M0001227` PPI, `M0001428` 社零当月同比,
`M0000162` FED, `M5525763` 社融存量同比. Browse the full catalog with
`cat data/macro_indicators.md` or query unknown codes in the Wind terminal's EDB browser.

`wsd` (multi-security multi-indicator time-series): CODES = comma-joined Wind codes
(`000300.SH,600519.SH`, `CI005001.WI`..`CI005030.WI` CITIC L1); INDICATORS = comma-joined
(`close,open`, `netprofit_ttm`, `pe_ttm`, `or_ttm2`, ...). Useful for pulling TTM
financials across many indices/stocks at once.

Notes: WindPy prints a startup banner to stdout (from the C library, can't be
suppressed in-process) — pipe through `tail -n +7` or use `--csv` + redirect.
`Fill=Previous` fills step-series (rates) forward to avoid sparse rows. EDB monthly
indicators put the Jan–Feb combined value in February (no January point), matching
the statistics bureau. American macro (FRED: US yields, credit spreads, Fed balance
sheet) is NOT in this skill — see `MidTermReport/ai_monitor/crawlers/l1_macro.py`.

---

## 4. FRED — US macro (rates / credit spreads / Fed balance sheet)

```bash
python3 ~/.zcode/skills/quant-data/fred.py ping                  # connectivity
python3 ~/.zcode/skills/quant-data/fred.py series                # list curated catalog
python3 ~/.zcode/skills/quant-data/fred.py fetch SERIES [--from YYYY-MM-DD] [--to ..] [--csv]
python3 ~/.zcode/skills/quant-data/fred.py bundle NAME [--from ..] [--to ..] [--csv]
```

FRED (St. Louis Fed) CSV endpoint — free, no API key, no login. Any python3 works
(no WindPy/oracle deps). Fetches try `requests` → `urllib` → `curl` (most robust
under VPN). Covers US Treasuries (2Y/10Y), Fed funds, breakeven inflation, credit
spreads (HY/IG OAS, AAA/BAA corporates), and the Fed balance sheet — series that
Wind EDB and iFinD don't surface cleanly. Full catalog: `data/macro_indicators.md`.

Bundles: `rates` (DGS10/DGS2/FEDFUNDS/T10YIE), `credit` (HY OAS/IG OAS/AAA/BAA),
`fed_bs` (WALCL), `all` (9 series). Or `fetch` arbitrary FRED series ids
comma-joined (any series from https://fred.stlouisfed.org works, e.g. `CPIAUCSL`
CPI, `UNRATE` unemployment).

Notes: FRED marks missing values as `.` → empty cells. Daily series have no
weekend/holiday rows; align-by-date means a row exists if ANY requested series
has a value that day.

---

## 5. iFinD — Tonghuashun HTTP API (auto token refresh)

```bash
/usr/bin/python3 ~/.zcode/skills/quant-data/ifind.py token        # force-refresh access_token, print it
/usr/bin/python3 ~/.zcode/skills/quant-data/ifind.py smoke        # connectivity: refresh + one realtime quote
/usr/bin/python3 ~/.zcode/skills/quant-data/ifind.py realtime --codes 300033.SZ [--indicators latest]
/usr/bin/python3 ~/.zcode/skills/quant-data/ifind.py history --codes 000001.SZ,600000.SH \
    --indicators open,high,low,close --startdate 2021-07-05 --enddate 2022-07-05
/usr/bin/python3 ~/.zcode/skills/quant-data/ifind.py edb --indicators M001620326 --startdate 2020-01-01 --enddate 2022-12-31
/usr/bin/python3 ~/.zcode/skills/quant-data/ifind.py post ENDPOINT --json '{"...":"..."}'   # generic
```

Requires `requests` — if missing, install: `/usr/bin/python3 -m pip install requests`.
Full endpoint reference: `data/ifind_query_manual.md` (realtime, history, high-freq,
basic_data_service, date_sequence, snap_shot, edb_service, data_pool, report_query,
smart_stock_picking, get_trade_dates).

Codes: `300033.SZ`, `000001.SZ`, `600000.SH`, `000300.SH`; comma-join multiples.

### Token refresh (automatic)
- `access_token` (short-lived) is sent in the request header.
- On `errorcode` `-1302`/`-1303` (expired) the client calls `/get_access_token`
  with the long-lived `refresh_token`, retries once, and **writes the new
  access_token back to `ifind.env`** for other processes.
- `refresh_token` is currently valid until **2026-12-31**. When it expires,
  get a new one from the iFinD **超级命令 (Super Command)** account page
  (https://quantapi.10jqka.com.cn) and paste it into `ifind.env`. Nothing else
  needs to change.

---

## Safety
- All `*.env` files are chmod 600 and contain credentials — never echo them or
  send them externally.
- WDS and Factor DB connectors are strictly read-only (reject DML/DDL).
- iFinD is an external API call subject to the account's permissions.
- This skill does not modify any existing skill or the original data files;
  `tools/` scripts rebuild `data/` artifacts reproducibly.

## Rebuilding the dictionaries (rarely needed)
```bash
/usr/bin/python3 ~/.zcode/skills/quant-data/tools/clean_wds_dict.py
/usr/bin/python3 ~/.zcode/skills/quant-data/tools/build_factor_schema.py
```
Run these only when the source WDS dictionary JSON or the Guosheng PDF changes.
