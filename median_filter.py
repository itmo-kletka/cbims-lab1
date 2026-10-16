import numpy as np
from PIL import Image, ImageFilter


def validate_kernel_size(kernel_size):
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
        raise TypeError("размер ядра должен быть целым числом")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("размер ядра должен быть положительным нечётным числом")
    return kernel_size


def validate_image(image):
    if not isinstance(image, np.ndarray):
        raise TypeError("изображение должно быть массивом numpy")
    if image.dtype != np.uint8:
        raise TypeError("изображение должно иметь тип uint8")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError("ожидается серое изображение HxW или цветное HxWx3")


def median_value(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _filter_channel(channel, kernel_size):
    radius = kernel_size // 2
    padded = np.pad(channel, radius, mode="edge")
    height, width = channel.shape
    result = np.empty((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            window = []
            for dy in range(kernel_size):
                row = padded[y + dy]
                for dx in range(kernel_size):
                    window.append(int(row[x + dx]))
            result[y, x] = median_value(window)
    return result


def median_filter_native(image, kernel_size):
    image = validate_image(image)
    kernel_size = validate_kernel_size(kernel_size)
    if image.ndim == 2:
        return _filter_channel(image, kernel_size)

    channels = [
        _filter_channel(image[:, :, channel], kernel_size) for channel in range(3)
    ]
    return np.stack(channels, axis=2)


def median_filter_pillow(image, kernel_size):
    image = validate_image(image)
    kernel_size = validate_kernel_size(kernel_size)
    mode = "L" if image.ndim == 2 else "RGB"
    filtered = Image.fromarray(image, mode=mode).filter(
        ImageFilter.MedianFilter(size=kernel_size)
    )
    return np.asarray(filtered)
