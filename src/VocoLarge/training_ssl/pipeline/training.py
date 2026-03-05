from src.VocoLarge.third_party_voco_large.utils.ops import concat_image
from src.VocoLarge.training_ssl.pipeline import forward_loss, unpack_voco_output, to_device


def train_one_batch(model, opt, scaler, batch, device):
    img, labels, crops = batch
    img, crops = concat_image(img), concat_image(crops)
    img, crops, labels = to_device(img, crops, labels, device)
    model.train()
    opt.zero_grad(set_to_none=True)

    loss = forward_loss(model, img, crops, labels, use_amp=True)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    return loss