from copy import deepcopy

from monai.transforms import (
    Compose,
    MapTransform,
    Randomizable,
    SpatialPadd,
    CenterSpatialCropd,
    RandCropByLabelClassesd,
)


class RandFullOrLabelCropd(Randomizable, MapTransform):
    """
    Randomly returns either:
      1. a full-field crop, e.g. 512x512x64
      2. a foreground-centered crop, e.g. 192x192x64

    IMPORTANT:
    This can return different spatial sizes across iterations.
    Use batch_size=1 unless you use separate loaders / custom batching.
    """

    def __init__(
        self,
        keys,
        label_key,
        full_roi_size,
        patch_roi_size,
        prob_full,
        num_classes,
        ratios,
        num_patch_samples=1,
        allow_missing_keys=False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.label_key = label_key
        self.full_roi_size = full_roi_size
        self.patch_roi_size = patch_roi_size
        self.prob_full = prob_full

        self.full_transform = Compose(
            [
                SpatialPadd(keys=keys, spatial_size=full_roi_size),
                CenterSpatialCropd(keys=keys, roi_size=full_roi_size),
            ]
        )

        self.patch_transform = Compose(
            [
                SpatialPadd(keys=keys, spatial_size=patch_roi_size),
                RandCropByLabelClassesd(
                    keys=keys,
                    label_key=label_key,
                    spatial_size=patch_roi_size,
                    num_classes=num_classes,
                    ratios=ratios,
                    num_samples=num_patch_samples,
                ),
            ]
        )

    def randomize(self, data=None):
        self._do_full = self.R.random() < self.prob_full

    def __call__(self, data):
        self.randomize(data)

        if self._do_full:
            d = self.full_transform(deepcopy(data))

            # Return list to match RandCropByLabelClassesd behavior.
            return [d]

        return self.patch_transform(deepcopy(data))