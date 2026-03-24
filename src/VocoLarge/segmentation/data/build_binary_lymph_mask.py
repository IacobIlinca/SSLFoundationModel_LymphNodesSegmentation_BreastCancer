import json
from pathlib import Path
from typing import Sequence, List, Dict, Any

import nibabel as nib
import numpy as np
from monai.transforms import MapTransform


class BuildBinaryLymphMaskd(MapTransform):
    """
    Build a single binary lymph-node mask from a list of mask paths.

    Expected input:
        d["mask_paths"] = [path1, path2, path3, ...]

    Output:
        d["label"] = np.ndarray with values {0,1}
    """

    def __init__(
        self,
        mask_paths_key: str = "mask_paths",
        output_key: str = "label",
        lymph_terms_json: str = "",
        save_matched_paths_key: str | None = "matched_mask_paths",
        allow_missing_keys: bool = False,
        no_lymph_patients_log_file: str = None
    ):
        super().__init__(keys=[mask_paths_key], allow_missing_keys=allow_missing_keys)

        self.mask_paths_key = mask_paths_key
        self.no_lymph_patients_log_file = no_lymph_patients_log_file
        self.output_key = output_key
        self.save_matched_paths_key = save_matched_paths_key

        with open(lymph_terms_json, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            terms = data
        elif isinstance(data, dict) and "terms" in data:
            terms = data["terms"]
        else:
            raise ValueError(
                "Invalid lymph terms JSON format. "
                "Expected either a list or a dict with key 'terms'."
            )

        self.lymph_terms = [str(t).lower().strip() for t in terms if str(t).strip()]

        if not self.lymph_terms:
            raise ValueError("No lymph terms found in JSON.")

    def _is_lymph_mask(self, path: str) -> bool:
        name = Path(path).name.lower()
        return any(term in name for term in self.lymph_terms)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)

        mask_paths: Sequence[str] = d[self.mask_paths_key]
        if not mask_paths:
            raise ValueError(
                f"No mask paths found under key '{self.mask_paths_key}' for case "
                f"{d.get('case_id', '<unknown>')}"
            )

        matched_paths = [p for p in mask_paths if self._is_lymph_mask(p)]

        if len(matched_paths) == 0:
            case_id = d.get("case_id", "<unknown>")
            available = [Path(p).name for p in mask_paths]

            msg = (
                f"{case_id} | NO_LYMPH_MATCH | masks={available}\n"
            )

            if self.no_lymph_patients_log_file is not None:
                with open(self.no_lymph_patients_log_file, "a") as f:
                    f.write(msg)
            else:
                print("[WARN]", msg.strip())
        # Use first available mask as reference space
        ref_img = nib.load(mask_paths[0])
        ref_shape = ref_img.shape

        binary_mask = np.zeros(ref_shape, dtype=np.uint8)

        for p in matched_paths:
            arr = nib.load(p).get_fdata()

            if arr.shape != ref_shape:
                raise ValueError(
                    f"Shape mismatch in case {d.get('case_id', '<unknown>')}: "
                    f"mask {p} has shape {arr.shape}, expected {ref_shape}"
                )

            binary_mask[arr > 0] = 1

        d[self.output_key] = binary_mask

        if self.save_matched_paths_key is not None:
            d[self.save_matched_paths_key] = matched_paths

        return d