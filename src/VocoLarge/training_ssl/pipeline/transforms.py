from monai.transforms import Compose
from src.VocoLarge.third_party_voco_large.utils import data_trans, voco_trans


def build_transforms(voco_args, no_aug: bool) -> Compose:
    trans_list = data_trans.get_chest_trans(voco_args)

    if no_aug:
        for i, t in enumerate(trans_list):
            if t.__class__.__name__ == "VoCoAugmentation":
                trans_list[i] = voco_trans.VoCoAugmentation(voco_args, aug=False)
                break

    return Compose(trans_list)