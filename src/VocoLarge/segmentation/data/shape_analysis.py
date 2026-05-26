import json
import os
import numpy as np
import matplotlib.pyplot as plt

JSON_PATH = "../data_audit_results/audit_workshop_clinically_delineated.json"
OUT_DIR = "../data_audit_results/shape_summary_output_workshop_clinically_delineated"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    images = data["images"]
    shapes = np.array([img["shape"] for img in images], dtype=np.int32)

    x = shapes[:, 0]
    y = shapes[:, 1]
    z = shapes[:, 2]

    def get_summary(arr):
        return {
            "min": int(arr.min()),
            "max": int(arr.max()),
            "average": float(arr.mean())
        }

    summary = {
        "num_images": len(images),
        "x": get_summary(x),
        "y": get_summary(y),
        "z": get_summary(z),
    }

    print(f"Number of images: {summary['num_images']}\n")
    for dim in ["x", "y", "z"]:
        print(f"{dim.upper()}:")
        print(f"  min     = {summary[dim]['min']}")
        print(f"  max     = {summary[dim]['max']}")
        print(f"  average = {summary[dim]['average']:.2f}")
        print()

    with open(os.path.join(OUT_DIR, "shape_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    def save_hist(arr, name):
        plt.figure(figsize=(8, 5))
        plt.hist(arr, bins=30)
        plt.title(f"Histogram of {name.upper()} dimension")
        plt.xlabel(name.upper())
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{name}_hist.png"), dpi=200)
        plt.close()

    save_hist(x, "x")
    save_hist(y, "y")
    save_hist(z, "z")

    print(f"Saved summary JSON and histograms in: {OUT_DIR}")


if __name__ == "__main__":
    main()