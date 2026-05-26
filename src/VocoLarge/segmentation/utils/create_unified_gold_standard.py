from pathlib import Path
import shutil
import re

# Change this if needed:
# If you run the script inside the folder that contains Images/ and DBCG_Workshop_Original...
ROOT = Path(".")

IMAGES_DIR = ROOT / "/mnt/data/ilinca/GS/Images"
GS_DIR = ROOT / "/mnt/data/ilinca/GS/DBCG_Workshop_Original_GoldStandard_nii_STAPLE_15_02"
OUT_DIR = ROOT / "/mnt/data/ilinca/unified_gold_standard"

OUT_DIR.mkdir(exist_ok=True)

def get_patient_id(folder_name: str) -> str | None:
    """
    Extracts patient id like pt5, pt10, pt123 from folder names such as:
    - pt10
    - DBCG_workshop_pt10
    """
    match = re.search(r"pt\d+", folder_name, re.IGNORECASE)
    return match.group(0).lower() if match else None


# Build mapping: pt10 -> Images/DBCG_workshop_pt10/image.nii.gz
image_map = {}

for img_patient_dir in IMAGES_DIR.iterdir():
    if not img_patient_dir.is_dir():
        continue

    patient_id = get_patient_id(img_patient_dir.name)
    if patient_id is None:
        continue

    image_file = img_patient_dir / "image.nii.gz"

    if image_file.exists():
        image_map[patient_id] = image_file
    else:
        print(f"Warning: no image.nii.gz found in {img_patient_dir}")


# Loop through gold-standard folders: pt5, pt10, ...
for gs_patient_dir in GS_DIR.iterdir():
    if not gs_patient_dir.is_dir():
        continue

    patient_id = get_patient_id(gs_patient_dir.name)
    if patient_id is None:
        print(f"Skipping folder with no patient id: {gs_patient_dir}")
        continue

    if patient_id not in image_map:
        print(f"Warning: no matching image found for {patient_id}")
        continue

    out_patient_dir = OUT_DIR / patient_id
    out_patient_dir.mkdir(exist_ok=True)

    # Copy image.nii.gz
    shutil.copy2(image_map[patient_id], out_patient_dir / "image.nii.gz")

    # Copy all files/folders from the gold-standard patient folder
    for item in gs_patient_dir.iterdir():
        dest = out_patient_dir / item.name

        if item.is_file():
            shutil.copy2(item, dest)

        elif item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)

    print(f"Created unified folder for {patient_id}")

print("Done.")