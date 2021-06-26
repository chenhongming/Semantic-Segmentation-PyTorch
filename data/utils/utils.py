from PIL import ImageOps


# fix batch size > 1 for training using torch.utils.data.DataLoader
def unified_size(image, mask, crop_size, padding=(0, 0, 0), ignore_label=255):
    w, h = image.size
    if isinstance(crop_size, int):
        _size = [crop_size, crop_size]
    elif isinstance(crop_size, list) or isinstance(crop_size, tuple):
        _size = crop_size
    else:
        raise RuntimeError("size error.")
    crop_w, crop_h = _size
    pad_w = max((crop_w - w), 0)
    pad_h = max((crop_h - h), 0)
    if pad_h > 0 or pad_w > 0:
        border = (pad_w // 2, pad_h // 2, (crop_w - w) - pad_w // 2, (crop_h - h) - pad_h // 2)
        image = ImageOps.expand(image, border=border, fill=padding)
        mask = ImageOps.expand(mask, border=border, fill=ignore_label)
    else:
        w_off = (w - crop_w) // 2
        h_off = (h - crop_h) // 2
        border = (w_off, h_off, (crop_w - w) - w_off, (crop_h - h) - h_off)
        image = ImageOps.crop(image, border=border)
        mask = ImageOps.crop(mask, border=border)
    return image, mask
