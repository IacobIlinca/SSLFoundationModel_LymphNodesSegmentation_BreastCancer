import os
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict

# ===================== EDIT THESE =====================
root_dir = "/mnt/data/ilinca/structured_cases_14_16"
output_csv = "src/VocoLarge/segmentation/masks/mask_name_counts.csv"
output_json = "src/VocoLarge/segmentation/masks/mask_name_counts.json"

output_review_csv = "src/VocoLarge/segmentation/masks/mask_name_review.csv"
output_review_json = "src/VocoLarge/segmentation/masks/mask_name_review.json"

LYMPH_TERMS_JSON = "src/VocoLarge/segmentation/masks/lymph_terms.json"
NOT_LYMPH_TERMS_JSON = "src/VocoLarge/segmentation/masks/not_lymph_terms.json"

allowed_exts = {".nii", ".nii.gz", ".nrrd", ".mha", ".mhd"}

ignore_if_contains = []


def has_valid_extension(filename: str) -> bool:
    fname = filename.lower()
    return any(fname.endswith(ext) for ext in allowed_exts)

def load_terms(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "terms" in data:
        return [t.lower() for t in data["terms"]]

    elif isinstance(data, list):
        return [t.lower() for t in data]

    else:
        raise ValueError(f"Invalid format in {path}")


def strip_medical_extension(filename: str) -> str:
    fname = filename
    if fname.lower().endswith(".nii.gz"):
        return fname[:-7]
    return Path(fname).stem


def normalize_mask_name(name: str) -> str:
    name = name.lower().strip()
    name = name.replace(" ", "_").replace("-", "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name


def should_ignore(stem: str) -> bool:
    stem_l = stem.lower()
    return any(term in stem_l for term in ignore_if_contains)


def find_matching_terms(mask_name: str, terms: list[str]) -> list[str]:
    return [term for term in terms if term in mask_name]


def classify_mask(mask_name: str, lymph_terms: list[str], not_lymph_terms: list[str]):
    lymph_hits = find_matching_terms(mask_name, lymph_terms)
    not_lymph_hits = find_matching_terms(mask_name, not_lymph_terms)

    if lymph_hits and not not_lymph_hits:
        category = "lymph_nodes"
    elif not_lymph_hits and not lymph_hits:
        category = "not_lymph_nodes"
    elif lymph_hits and not_lymph_hits:
        category = "conflict"
    else:
        category = "unknown"

    return category, lymph_hits, not_lymph_hits


def main():
    counts = Counter()
    examples = defaultdict(list)
    patient_examples = defaultdict(list)

    lymph_include_terms = load_terms(LYMPH_TERMS_JSON)
    not_lymph_terms = load_terms(NOT_LYMPH_TERMS_JSON)

    total_patients = 0
    total_mask_files = 0

    for patient_name in sorted(os.listdir(root_dir)):
        patient_path = os.path.join(root_dir, patient_name)

        if not os.path.isdir(patient_path):
            continue

        total_patients += 1

        for filename in os.listdir(patient_path):
            full_path = os.path.join(patient_path, filename)

            if not os.path.isfile(full_path):
                continue

            if not has_valid_extension(filename):
                continue

            stem = strip_medical_extension(filename)

            if should_ignore(stem):
                continue

            norm_name = normalize_mask_name(stem)

            counts[norm_name] += 1
            total_mask_files += 1

            if len(examples[norm_name]) < 5:
                examples[norm_name].append(full_path)

            if len(patient_examples[norm_name]) < 10:
                patient_examples[norm_name].append(patient_name)

    # original summary csv
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mask_name", "count", "example_patients", "example_paths"])
        for mask_name, count in counts.most_common():
            writer.writerow([
                mask_name,
                count,
                " | ".join(patient_examples[mask_name]),
                " | ".join(examples[mask_name]),
            ])

    # original summary json
    json_data = {
        mask_name: {
            "count": count,
            "example_patients": patient_examples[mask_name],
            "example_paths": examples[mask_name],
        }
        for mask_name, count in counts.most_common()
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    # categorized review
    review_rows = []
    category_counts = Counter()

    for mask_name, count in counts.most_common():
        category, lymph_hits, not_lymph_hits = classify_mask(
            mask_name,
            lymph_include_terms,
            not_lymph_terms
        )
        category_counts[category] += 1

        review_rows.append({
            "mask_name": mask_name,
            "count": count,
            "category": category,
            "matched_lymph_terms": lymph_hits,
            "matched_not_lymph_terms": not_lymph_hits,
            "example_patients": patient_examples[mask_name],
            "example_paths": examples[mask_name],
        })

    with open(output_review_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mask_name",
            "count",
            "category",
            "matched_lymph_terms",
            "matched_not_lymph_terms",
            "example_patients",
            "example_paths",
        ])
        for row in review_rows:
            writer.writerow([
                row["mask_name"],
                row["count"],
                row["category"],
                " | ".join(row["matched_lymph_terms"]),
                " | ".join(row["matched_not_lymph_terms"]),
                " | ".join(row["example_patients"]),
                " | ".join(row["example_paths"]),
            ])

    with open(output_review_json, "w", encoding="utf-8") as f:
        json.dump(review_rows, f, indent=2)

    print(f"[INFO] Total patients: {total_patients}")
    print(f"[INFO] Total mask files counted: {total_mask_files}")
    print(f"[INFO] Unique mask names found: {len(counts)}")
    print(f"[INFO] CSV saved to: {output_csv}")
    print(f"[INFO] JSON saved to: {output_json}")
    print(f"[INFO] Review CSV saved to: {output_review_csv}")
    print(f"[INFO] Review JSON saved to: {output_review_json}")

    print("\n[INFO] Category summary:")
    for category, n in category_counts.items():
        print(f"  {category}: {n}")

    print("\n[INFO] Top 50 mask names:")
    for mask_name, count in counts.most_common(50):
        print(f"{mask_name}: {count}")


if __name__ == "__main__":
    main()