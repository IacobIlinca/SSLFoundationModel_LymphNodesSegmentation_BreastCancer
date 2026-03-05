from dataclasses import dataclass

@dataclass
class VocoArgs:
    in_channels: int = 1
    feature_size: int = 48
    dropout_path_rate: float = 0.0
    use_checkpoint: bool = True
    spatial_dims: int = 3

    roi_x: int = 96
    roi_y: int = 96
    roi_z: int = 64

    space_x: float = 1.5
    space_y: float = 1.5
    space_z: float = 1.5

    local_rank: int = 0
    sw_batch_size: int = 1
    amp: bool = True
    device: str = "cuda"


def build_voco_args(
    roi_x: int,
    roi_y: int,
    roi_z: int,
    device: str,
    feature_size: int = 48,
    use_checkpoint: bool = True,
    amp: bool = True,
    sw_batch_size: int = 1,
) -> VocoArgs:
    return VocoArgs(
        roi_x=roi_x, roi_y=roi_y, roi_z=roi_z,
        device=device,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
        amp=amp,
        sw_batch_size=sw_batch_size,
    )