import pandas as pd
from pathlib import Path

csv_path = Path(
    "src/VocoLarge/segmentation/masks/binary_mask_selection_audit/required_multiclass_nodes_summary2.csv"
)

df = pd.read_csv(csv_path).fillna("")

labels = [
    "level1",
    "level2",
    "level3",
    "level4",
    "interpectoral",
    "imn",
]

# A label is present if n_<label>_masks > 0
for label in labels:
    col = f"n_{label}_masks"
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")
    df[f"has_{label}"] = df[col].astype(int) > 0


def count_cases(required_labels):
    mask = pd.Series(True, index=df.index)

    for label in required_labels:
        mask &= df[f"has_{label}"]

    return int(mask.sum()), df.loc[mask, "case_id"].tolist()


scenarios = {}

# Current strict rule: all 6 required
scenarios["all_6_required"] = labels

# One optional at a time
for optional_label in labels:
    required = [label for label in labels if label != optional_label]
    scenarios[f"{optional_label}_optional"] = required

# Level1 + level4 optional together
scenarios["level1_and_level4_optional"] = [
    label for label in labels
    if label not in ["level1", "level4"]
]

# Custom minimal required-label scenarios
scenarios["only_level2_and_imn_required"] = [
    "level2",
    "imn",
]

scenarios["only_level2_and_level3_required"] = [
    "level2",
    "level3",
]

rows = []

out_dir = csv_path.parent / "optional_label_analysis"
out_dir.mkdir(exist_ok=True)

for scenario_name, required_labels in scenarios.items():
    n_cases, case_ids = count_cases(required_labels)

    optional_labels = [
        label for label in labels
        if label not in required_labels
    ]

    rows.append({
        "scenario": scenario_name,
        "n_cases": n_cases,
        "required_labels": "|".join(required_labels),
        "optional_labels": "|".join(optional_labels),
    })

    with open(out_dir / f"{scenario_name}_case_ids.txt", "w") as f:
        for case_id in case_ids:
            f.write(str(case_id) + "\n")

summary = pd.DataFrame(rows)
summary = summary.sort_values("n_cases", ascending=False)

summary_path = out_dir / "optional_label_counts_summary.csv"
summary.to_csv(summary_path, index=False)

print(summary.to_string(index=False))
print()
print(f"Saved summary to: {summary_path}")
print(f"Saved case ID files to: {out_dir}")