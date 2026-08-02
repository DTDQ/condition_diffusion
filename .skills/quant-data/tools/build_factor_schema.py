#!/usr/bin/env python3
"""
Build the factor-DB schema (factor + market schemas) by introspecting the LIVE
MySQL database via information_schema, then enrich each field with the Chinese
meaning from the Guosheng PDF list where available.

The live DB is the source of truth (table list, field names, types). The PDF
(国盛证券金融工程数据库清单.pdf, 2018-07-26) supplies older Chinese field notes;
fields that exist in the DB but not the PDF get an empty `cn`, and PDF-only
fields are dropped. This keeps the schema accurate as the team evolves the DB.

Reads:  live MySQL at 222.66.94.9:33060 (factors + market + mutualfunddata + wyckoff_tech)
        /Users/erleye/Documents/工作资料/国盛证券/国盛证券金融工程数据库清单.pdf  (cn notes)
Writes: ../data/factor_schema.json   (db -> table -> fields[{name,type,cn,note}])
        ../data/factor_schema.md     (human-readable)

Idempotent. Run:
    python3 build_factor_schema.py
"""
import json
import os
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # so we can import factor.py's connector
import factor as _factor  # noqa: E402

OUT_JSON = _HERE.parent / "data" / "factor_schema.json"
OUT_MD = _HERE.parent / "data" / "factor_schema.md"
PDF = Path("/Users/erleye/Documents/工作资料/国盛证券/国盛证券金融工程数据库清单.pdf")

# Schemas to document (the user-facing quant schemas; skip information_schema/mysql/sys/...).
TARGET_SCHEMAS = ["market", "factors", "mutualfunddata", "wyckoff_tech"]
SCHEMA_TITLES = {
    "market": "中国A股基础数据库",
    "factors": "中国A股因子数据库 (Barra/Alpha)",
    "mutualfunddata": "公募基金数据库 (收益/载荷分解)",
    "wyckoff_tech": "威科夫技术指标库",
}


# --------------------------------------------------------------------------- #
# live schema introspection
# --------------------------------------------------------------------------- #
def introspect() -> dict:
    """Query information_schema for tables + columns of each target schema."""
    conn = _factor.get_connection()
    schema: dict = {}
    try:
        cur = conn.cursor()
        for db in TARGET_SCHEMAS:
            cur.execute(
                "SELECT TABLE_NAME, TABLE_COMMENT FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA=%s ORDER BY TABLE_NAME",
                (db,),
            )
            tables = cur.fetchall()
            schema[db] = {"title": SCHEMA_TITLES.get(db, db), "tables": {}}
            for tname, tcomment in tables:
                cur.execute(
                    "SELECT COLUMN_NAME, COLUMN_TYPE, COLUMN_COMMENT, IS_NULLABLE, "
                    "COLUMN_KEY, COLUMN_DEFAULT, EXTRA "
                    "FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s ORDER BY ORDINAL_POSITION",
                    (db, tname),
                )
                cols = cur.fetchall()
                schema[db]["tables"][tname] = {
                    "comment": tcomment or "",
                    "fields": [
                        {
                            "name": c[0],
                            "type": c[1],
                            "cn": c[2] or "",     # live COLUMN_COMMENT (usually empty here)
                            "nullable": c[3],
                            "key": c[4],
                            "default": str(c[5]) if c[5] is not None else None,
                            "extra": c[6] or "",
                        }
                        for c in cols
                    ],
                }
        cur.close()
    finally:
        conn.close()
    return schema


# --------------------------------------------------------------------------- #
# PDF cn-note enrichment
# --------------------------------------------------------------------------- #
def load_pdf_cn_notes() -> dict:
    """Parse the Guosheng PDF into {table_name: {field_name: cn}}.

    Used only to enrich the live schema with Chinese notes for the older tables
    that still exist. Tables/fields not in the live DB are simply ignored.
    """
    try:
        import pdfplumber
    except ImportError:
        print("[build_factor_schema] pdfplumber not installed; skipping PDF enrichment",
              file=sys.stderr)
        return {}
    if not PDF.exists():
        print(f"[build_factor_schema] PDF not found, skipping enrichment: {PDF}",
              file=sys.stderr)
        return {}

    SECTION_RE = re.compile(r"^(\d+)\.(\d+)\s+(.+?)\s+([a-z][a-z0-9_]*)\s*$")
    FIELD_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
    TYPES = {"bigint", "int", "double", "date", "timestamp", "json", "varchar"}

    notes: dict = {}
    current_table = None
    with pdfplumber.open(PDF) as pdf:
        for page in pdf.pages:
            for raw in (page.extract_text() or "").splitlines():
                line = raw.strip()
                if not line or re.match(r"^P\.\d+$", line):
                    continue
                if line.startswith("国盛证券金融工程") or line.startswith("2018年"):
                    continue
                m = SECTION_RE.match(line)
                if m:
                    current_table = m.group(4)
                    notes.setdefault(current_table, {})
                    continue
                if current_table:
                    toks = line.split()
                    if len(toks) >= 2 and FIELD_NAME_RE.match(toks[0]):
                        typ = toks[1].lower()
                        if any(typ == t or typ.startswith(t + "(") for t in TYPES):
                            cn = " ".join(toks[2:]).strip()
                            notes[current_table][toks[0]] = cn
    return notes


def enrich(schema: dict, notes: dict) -> None:
    """Fill in `cn` from PDF notes where the live COLUMN_COMMENT is empty."""
    for db in schema.values():
        for tname, t in db["tables"].items():
            pdf_fields = notes.get(tname, {})
            for f in t["fields"]:
                if not f["cn"] and f["name"] in pdf_fields:
                    f["cn"] = pdf_fields[f["name"]]


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def write_outputs(schema: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 因子数据库结构 (factors + market + mutualfunddata + wyckoff_tech)", "",
             "来源: 实时 introspection of MySQL 222.66.94.9:33060 (live, authoritative) "
             "+ 国盛证券金融工程数据库清单.pdf (Chinese field notes, 2018-07-26).", "",
             "> 库结构以实时数据库为准；中文释义优先取 live COLUMN_COMMENT，缺失时回退到 PDF。",
             ""]
    for db_name, db in schema.items():
        lines.append(f"## 数据库 `{db_name}` — {db['title']}")
        lines.append("")
        lines.append(f"共 {len(db['tables'])} 张表。")
        lines.append("")
        # table index
        lines.append("| 表名 | 字段数 | 说明 |")
        lines.append("| --- | --- | --- |")
        for tname, t in sorted(db["tables"].items()):
            cmt = (t.get("comment") or "").replace("|", "\\|")
            lines.append(f"| `{tname}` | {len(t['fields'])} | {cmt} |")
        lines.append("")
        # full field lists
        for tname, t in sorted(db["tables"].items()):
            lines.append(f"### `{db_name}.{tname}`  ({len(t['fields'])} 字段)")
            if t.get("comment"):
                lines.append(f"*{t['comment']}*")
            lines.append("")
            lines.append("| 字段 | 类型 | 含义 | 可空 | 键 |")
            lines.append("| --- | --- | --- | --- | --- |")
            for f in t["fields"]:
                cn = (f["cn"] or "").replace("|", "\\|")
                lines.append(f"| `{f['name']}` | {f['type']} | {cn} | {f['nullable']} | {f['key']} |")
            lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    schema = introspect()
    notes = load_pdf_cn_notes()
    enrich(schema, notes)
    write_outputs(schema)
    n_tables = sum(len(db["tables"]) for db in schema.values())
    n_fields = sum(len(t["fields"]) for db in schema.values() for t in db["tables"].values())
    print(f"[build_factor_schema] OK: {len(schema)} dbs, {n_tables} tables, {n_fields} fields")
    for db_name, db in schema.items():
        print(f"  {db_name}: {db['title']} — {len(db['tables'])} tables")
    print(f"[build_factor_schema] wrote {OUT_JSON}")
    print(f"[build_factor_schema] wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
