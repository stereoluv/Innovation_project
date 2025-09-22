

import json
import re

# --- CONFIG ---
INPUT_PATH = "basic_data_3.jsonl"          # original file
OUTPUT_PATH = "basic_data_3.cleaned.jsonl" # output JSONL with unique ids
# -----------------------------------

def iter_objects(raw: str):
    """
    Yields JSON object strings from:
      - concatenated objects split on '}\n{'
    """
    parts = re.split(r"}\s*\n\s*{", raw.strip())
    for p in parts:
        if not p.startswith("{"):
            p = "{" + p
        if not p.endswith("}"):
            p = p + "}"
        yield p

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        raw = f.read()

    seen = set()
    decode_errors = 0
    dup_skipped = 0
    written = 0
    total_split = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for total_split, obj_str in enumerate(iter_objects(raw), start=1):
            try:
                rec = json.loads(obj_str)
            except json.JSONDecodeError:
                decode_errors += 1
                continue

            id = rec.get("id")
            if isinstance(id, str) and id in seen:
                dup_skipped += 1
                continue
            if isinstance(id, str):
                seen.add(id)

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print("Cleaning summary:")
    print(f"  # of input objects: {total_split}")
    print(f"  JSON decode errors skipped: {decode_errors}")
    print(f"  Duplicate IDs skipped: {dup_skipped}")
    print(f"  Records kept: {written}")
    print(f"  Output: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
