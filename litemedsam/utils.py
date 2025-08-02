from PIL import Image
import numpy as np

def preprocess_image_and_box(image, box, target_size=256):
    original_size = image.size
    ratio = target_size / max(original_size)
    new_size = tuple([int(x * ratio) for x in original_size])
    image = image.resize(new_size, Image.BILINEAR)

    pad_image = Image.new("RGB", (target_size, target_size))
    pad_image.paste(image, ((target_size - new_size[0]) // 2, (target_size - new_size[1]) // 2))

    # Adjust box
    x1, y1, x2, y2 = box
    x1 = int(x1 * ratio + (target_size - new_size[0]) // 2)
    x2 = int(x2 * ratio + (target_size - new_size[0]) // 2)
    y1 = int(y1 * ratio + (target_size - new_size[1]) // 2)
    y2 = int(y2 * ratio + (target_size - new_size[1]) // 2)

    return pad_image, [x1, y1, x2, y2], original_size

def overlay_mask_on_image(image, mask):
    image_np = np.array(image).copy()
    mask_colored = np.zeros_like(image_np)
    mask_colored[:, :, 1] = mask  # Green channel

    alpha = 0.5
    overlayed = (image_np * (1 - alpha) + mask_colored * alpha).astype(np.uint8)
    return overlayed
