import csv
import os
import shutil
from pathlib import Path
import re


# ====== EDIT THESE ======
csv_file = None
txt_file = "/mnt/data/flaviu/workshop_not_anon_id_12_03.txt"

source_root = "/mnt/data/flaviu/rtnation_02_02"
target_root = "/mnt/data/ilinca/workshop_12_03"

id_column = 0
has_header = True
# ========================


source_root = Path(source_root)
target_root = Path(target_root)
target_root.mkdir(parents=True, exist_ok=True)

copied = []
missing = []
skipped = []


def copy_case(case_id, row_info=""):
    """Copy one patient folder."""
    case_id = case_id.strip()
    if not case_id:
        return

    src_case_dir = source_root / case_id
    dst_case_dir = target_root / case_id

    if not src_case_dir.exists() or not src_case_dir.is_dir():
        print(f"[MISSING] {row_info}{case_id}")
        missing.append(case_id)
        return

    if dst_case_dir.exists():
        print(f"[SKIP] already exists: {case_id}")
        skipped.append(case_id)
        return

    shutil.copytree(src_case_dir, dst_case_dir)
    print(f"[COPIED] {case_id}")
    copied.append(case_id)


def process_csv(csv_path):
    """Read IDs from CSV."""
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)

        if has_header:
            next(reader, None)

        for row_num, row in enumerate(reader, start=2 if has_header else 1):
            if not row:
                continue

            case_id = row[id_column]
            copy_case(case_id, f"row {row_num}: ")




def process_txt(txt_path):
    """Read IDs from TXT where IDs may be comma or whitespace separated."""
    with open(txt_path, "r", encoding="utf-8-sig") as f:
        for line_num, line in enumerate(f, start=1):

            ids = re.split(r"[,\s]+", line.strip())

            for case_id in ids:
                if not case_id:
                    continue

                copy_case(case_id, f"line {line_num}: ")


# ===== Run whichever input exists =====

if csv_file:
    process_csv(csv_file)

if txt_file:
    process_txt(txt_file)


# ===== Summary =====

print("\n=== DONE ===")
print(f"Copied : {len(copied)}")
print(f"Missing: {len(missing)}")
print(f"Skipped: {len(skipped)}")

if missing:
    print("\nMissing IDs:")
    for x in missing:
        print(x)