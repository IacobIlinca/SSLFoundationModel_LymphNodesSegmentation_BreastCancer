import json
import nibabel as nib
import numpy as np

JSON_PATH = "../data_audit_results/audit_structured_14_16.json"

with open(JSON_PATH, "r") as f:
    data = json.load(f)

images = data["images"]

rows = []
for item in images:
    path = item["path"]
    shape = item["shape"]  # [x, y, z]

    nii = nib.load(path)
    spacing = nii.header.get_zooms()[:3]  # (sx, sy, sz)

    fov = (
        shape[0] * spacing[0],
        shape[1] * spacing[1],
        shape[2] * spacing[2],
    )

    rows.append({
        "path": path,
        "shape": shape,
        "spacing": spacing,
        "fov_mm": fov,
    })

# print a few 512 and 1024 cases
for r in rows[:10]:
    print(r["path"])
    print("  shape   :", r["shape"])
    print("  spacing :", tuple(round(v, 3) for v in r["spacing"]))
    print("  fov_mm  :", tuple(round(v, 1) for v in r["fov_mm"]))
    print()

x_sizes = np.array([r["shape"][0] for r in rows])
print("Unique x sizes:", sorted(set(x_sizes.tolist())))