#!/usr/bin/env python3
"""
Clean the Wind WDS data dictionary JSON (parsed from PDFs with heavy watermark noise).

Reads:  /Users/erleye/Documents/Python/WDS/wds_dictionary_original.json  (151 tables, 16 cats)
Writes: ../data/wds_dictionary.json        (cleaned: categories -> table -> fields[{name,cn,type,fill_rate,desc}])
        ../data/wds_dictionary_index.md    (browsable index by category)

Idempotent: re-run anytime the source dict changes. Original file is never modified.

Run:
    python3 clean_wds_dict.py
"""
import json
import re
import sys
from pathlib import Path

SRC = Path("/Users/erleye/Documents/Python/WDS/wds_dictionary_original.json")
OUT_JSON = Path(__file__).resolve().parent.parent / "data" / "wds_dictionary.json"
OUT_MD = Path(__file__).resolve().parent.parent / "data" / "wds_dictionary_index.md"

# Watermark/noise tokens injected by the PDF parser. These appear interleaved with
# real text and must be stripped. They come from header/footer overlays like
# "F157990", "9010 0 0 3", "m.c o c", "yeerle", "gl @", "s.c o m".
NOISE_FRAGMENTS = [
    "yeerle", "yeerle@", "yeerl", "yeer",
    "m.c o", "m.c.o", "m.c", "mco", "m o c", "moc",
    "s.c o", "s.c.o", "s.c", "sco",
    "c o m", "c.o.m", "com",
    "o c m", "o m c",
    "gl @", "g l @", "@gl", "gle", "gl",
    "F157990", "F1579", "F15", "1579", "790", "9010",
    "9010 0 0 3", "9003", "900", "009", "09", "79 9", "799",
    "3 003", "003", "0 3", "0 0 3",
]
# Build a regex alternation, longest first so longer fragments win.
_NOISE_RE = re.compile(
    "(" + "|".join(re.escape(s) for s in sorted(NOISE_FRAGMENTS, key=len, reverse=True)) + ")",
    re.IGNORECASE,
)


def _strip_noise(text: str) -> str:
    """Remove watermark fragments and collapse whitespace/newlines."""
    if text is None:
        return ""
    s = str(text)
    s = _NOISE_RE.sub(" ", s)
    # Collapse all whitespace runs (incl. newlines) to a single space.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_field_name(raw: str) -> str:
    """Field names are UPPER_CASE identifiers.

    The PDF parser interleaves lowercase watermark letters and newlines into the
    name (e.g. ``S_INFO_WIND\\nnCODE`` -> ``S_INFO_WINDCODE``). Real WDS field
    names are exclusively uppercase letters, digits and underscore, so we keep
    only ``[A-Z0-9_]``: newlines drop out (re-joining the split name) and the
    stray lowercase ``n``/``o``/``c`` noise letters are discarded.
    """
    if raw is None:
        return ""
    return re.sub(r"[^A-Z0-9_]", "", str(raw))


_TYPE_RE = re.compile(r"(VARCHAR2|NUMBER|DATE|CHAR|CLOB|FLOAT|DOUBLE|LONG|RAW|TIMESTAMP)\s*\(?(\s*\d+\s*)?\)?", re.IGNORECASE)


def clean_type(raw: str) -> str:
    """Normalise a type string, e.g. 'VARCHAR2(\\n40)' -> 'VARCHAR2(40)', 'DOUBLE' stays."""
    if raw is None:
        return ""
    s = _strip_noise(raw)
    s = s.replace(" ", "")
    m = _TYPE_RE.search(s)
    if m:
        base = m.group(1).upper()
        length = m.group(2)
        if length:
            return f"{base}({length.strip()})"
        return base
    return s


def clean_fill_rate(raw: str) -> str:
    """Extract a 'NN.NN%' fill-rate value if present, else return cleaned text.

    The PDF noise injects single letters between the digits and the percent sign
    (e.g. ``100.00m%``). Strip a lone lowercase letter that sits between a digit
    and ``%``, then match a plausible percentage (integer part <= 100).
    """
    if raw is None:
        return ""
    s = _strip_noise(raw)
    s = re.sub(r"(?<=[\d.])([a-z])+(?=%)", "", s)
    for m in re.finditer(r"(\d{1,3})(?:\.(\d{1,4}))?\s*%", s):
        if int(m.group(1)) <= 100:
            whole, frac = m.group(1), m.group(2)
            return f"{whole}.{frac}%" if frac is not None else f"{whole}%"
    return s


# Lone watermark tokens that appear as standalone "words" inside cn/desc text.
# These are digits/letters injected by the PDF watermark layer (79, 9010, 003,
# m, o, c, n, e, l, @, ms, gl, ...). We only drop them when they stand alone as
# a whitespace-separated token, so real content like "UTF-8" or "FY1" is kept.
_WATERMARK_TOKENS = {
    "79", "79", "799", "7990", "9010", "9003", "900", "90", "09", "9", "09", "003",
    "03", "3", "003", "0", "0", "3", "1579", "790", "F15", "F1579", "F157990",
    "m", "o", "c", "n", "e", "l", "ms", "ms.", "m.", "o.", "c.", "n.", "gl",
    "@", "e.", "l.", "@gl", "gle", "yeerle", "yeer", "yeerl", "yeerle@",
    "c.o.m", "m.c.o", "s.c.o", "c.o", "m.c", "o.m", "c.m", "m.o",
    "o.c", "s.c", "m.o.c", "c.o.m", "o.m.c", "m.c.o", "s.c.o.m",
}


def clean_text(raw: str) -> str:
    """General free-text cleanup for chinese names / descriptions.

    Strategy: strip the known watermark fragments first (handles glued noise like
    ``100.00m%``), collapse whitespace, then drop standalone watermark tokens
    (lone digits/letters that the watermark layer injected between real words).
    Real tokens — multi-letter English (UTF, FY1), percentages, parenthesised
    notes, pure-Chinese phrases — are preserved.
    """
    s = _strip_noise(raw)
    if not s:
        return ""
    out = []
    for tok in s.split(" "):
        if tok == "":
            continue
        # Drop pure single-letter/pure-watermark tokens, but keep tokens that
        # mix chinese+latin (e.g. "公司ID") or contain digits as part of a real
        # word (e.g. "FY1", "UTF-8"). A token is watermark only if it is ENTIRELY
        # in the watermark set OR is a lone digit/letter.
        if tok in _WATERMARK_TOKENS:
            continue
        if re.fullmatch(r"[a-zA-Z]", tok):  # lone letter
            continue
        if re.fullmatch(r"\d{1,2}", tok) and tok not in {"10", "11", "12", "20", "21", "30", "60", "80"}:
            # short standalone number; keep only common real ones (months/days)
            continue
        out.append(tok)
    result = "".join(out)  # join without space: chinese has no spaces
    # Re-insert a space only where an ASCII token directly abuts chinese, for
    # readability of mixed-language values like "公司ID" -> keep as is.
    return result.strip()


def clean_field(f: dict) -> dict:
    return {
        "name": clean_field_name(f.get("字段名", "")),
        "cn": clean_text(f.get("字段中文名", "")),
        "type": clean_type(f.get("字段类型", "")),
        "fill_rate": clean_fill_rate(f.get("有值率", "")),
        "desc": clean_text(f.get("释义", "")),
    }


def is_noise_field(field: dict) -> bool:
    """A field with no usable name after cleaning is junk."""
    name = clean_field_name(field.get("字段名", ""))
    if not name:
        return True
    # Names that are purely numeric or a single letter are noise.
    if re.fullmatch(r"[0-9]+", name) or len(name) == 1:
        return True
    return False


def clean_table(tname: str, t: dict) -> dict:
    fields = [clean_field(f) for f in t.get("fields", []) if not is_noise_field(f)]
    # De-duplicate by field name (PDF noise sometimes duplicated rows), keep first.
    seen = set()
    deduped = []
    for f in fields:
        if f["name"] and f["name"] not in seen:
            seen.add(f["name"])
            deduped.append(f)
    return {
        "table_name": tname,
        "category": t.get("metadata", {}).get("所属模块：", "") or "",
        "description": clean_text(t.get("description", "")),
        "fields": deduped,
    }


def main() -> int:
    if not SRC.exists():
        print(f"[clean_wds_dict] source not found: {SRC}", file=sys.stderr)
        return 1
    raw = json.loads(SRC.read_text(encoding="utf-8"))

    cleaned = {}  # category -> { table_name -> table_obj }
    stats = {"categories": 0, "tables": 0, "fields": 0, "dropped_fields": 0}
    for cat, tables in raw.items():
        stats["categories"] += 1
        cleaned[cat] = {}
        for tname, t in tables.items():
            ct = clean_table(tname, t)
            stats["dropped_fields"] += len(t.get("fields", [])) - len(ct["fields"])
            cleaned[cat][tname] = ct
            stats["tables"] += 1
            stats["fields"] += len(ct["fields"])

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")

    # Browsable markdown index.
    lines = ["# WDS 数据字典索引", "", f"共 {stats['categories']} 个分类、{stats['tables']} 张表、{stats['fields']} 个字段。", ""]
    for cat, tables in cleaned.items():
        lines.append(f"## {cat}")
        lines.append("")
        lines.append("| 表名 | 字段数 | 关键字段 |")
        lines.append("| --- | --- | --- |")
        for tname, t in sorted(tables.items()):
            key = ", ".join(f["name"] for f in t["fields"][:4])
            lines.append(f"| `{tname}` | {len(t['fields'])} | {key} |")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"[clean_wds_dict] OK: {stats['categories']} cats, {stats['tables']} tables, "
          f"{stats['fields']} fields ({stats['dropped_fields']} dropped as noise)")
    print(f"[clean_wds_dict] wrote {OUT_JSON}")
    print(f"[clean_wds_dict] wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
