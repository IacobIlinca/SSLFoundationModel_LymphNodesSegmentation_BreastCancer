import json
import re
from pathlib import Path
from typing import Sequence, Dict, Any

import nibabel as nib
import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import MapTransform


class BuildBinaryLymphMaskd(MapTransform):
    """
    Build a single binary lymph-node mask from a list of mask paths.

    Expected input:
        d["image"]      = already loaded MONAI image (MetaTensor)
        d["mask_paths"] = [path1, path2, path3, ...]

    Output:
        d["label"] = MetaTensor with values {0,1}, carrying image metadata

    Matching rule:
        include mask if:
          - it matches at least one lymph/include term
          - and matches none of the exclude/not_lymph terms
    """

    def __init__(
        self,
        mask_paths_key: str = "mask_paths",
        image_key: str = "image",
        output_key: str = "label",
        lymph_terms_json: str = "",
        not_lymph_terms_json: str = "",
        save_matched_paths_key: str | None = "matched_mask_paths",
        allow_missing_keys: bool = False,
        no_lymph_patients_log_file: str | None = None,
    ):
        super().__init__(keys=[mask_paths_key, image_key], allow_missing_keys=allow_missing_keys)

        self.mask_paths_key = mask_paths_key
        self.image_key = image_key
        self.output_key = output_key
        self.save_matched_paths_key = save_matched_paths_key
        self.no_lymph_patients_log_file = no_lymph_patients_log_file
        self.size_pattern = re.compile(r"[0-9]+.*(mm|cm)")

        self.lymph_terms = self._load_terms(
            lymph_terms_json,
            expected_name="lymph_terms_json",
        )

        self.not_lymph_terms = []
        if not_lymph_terms_json:
            self.not_lymph_terms = self._load_terms(
                not_lymph_terms_json,
                expected_name="not_lymph_terms_json",
            )

    def _load_terms(self, json_path: str, expected_name: str) -> list[str]:
        with open(json_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            terms = data
        elif isinstance(data, dict) and "terms" in data:
            terms = data["terms"]
        else:
            raise ValueError(
                f"Invalid JSON format in {expected_name}: {json_path}. "
                f"Expected either a list or a dict with key 'terms'."
            )

        terms = [str(t).lower().strip() for t in terms if str(t).strip()]
        if not terms:
            raise ValueError(f"No terms found in {expected_name}: {json_path}")

        return terms

    def _is_lymph_mask(self, path: str) -> bool:
        name = Path(path).name.lower()

        has_include = any(term in name for term in self.lymph_terms)
        if not has_include:
            return False

        has_exclude = any(term in name for term in self.not_lymph_terms)
        if has_exclude:
            return False

        if self.size_pattern.search(name):
            return False

        return True

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)

        if self.image_key not in d:
            raise ValueError(
                f"'{self.image_key}' must already be loaded before BuildBinaryLymphMaskd runs."
            )

        image_mt = d[self.image_key]
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
            msg = f"{case_id} | NO_LYMPH_MATCH | masks={available}\n"

            if self.no_lymph_patients_log_file is not None:
                with open(self.no_lymph_patients_log_file, "a") as f:
                    f.write(msg)
            else:
                print("[WARN]", msg.strip())

        # Use image spatial shape as reference
        ref_shape = tuple(image_mt.shape[-3:])
        binary_mask = np.zeros(ref_shape, dtype=np.uint8)

        # Build union mask
        for p in matched_paths:
            arr = nib.load(p).get_fdata()

            if arr.shape != ref_shape:
                raise ValueError(
                    f"Shape mismatch in case {d.get('case_id', '<unknown>')}: "
                    f"mask {p} has shape {arr.shape}, expected {ref_shape}"
                )

            binary_mask[arr > 0] = 1

        # Convert to MetaTensor and copy image metadata
        label = MetaTensor(
            torch.as_tensor(binary_mask, dtype=torch.uint8),
            meta=image_mt.meta.copy() if hasattr(image_mt, "meta") else {},
        )

        if hasattr(image_mt, "affine"):
            label.affine = image_mt.affine.clone()

        d[self.output_key] = label

        if self.save_matched_paths_key is not None:
            d[self.save_matched_paths_key] = matched_paths

        return d