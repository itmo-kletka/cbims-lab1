import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from median_filter import median_filter_native, median_filter_pillow

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
IMAGE_PATH = Path(__file__).resolve().parent / "source" / "photo.jpg"


def load_image(path):
    if not path.is_file():
        raise FileNotFoundError(f"нет файла {path}")
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return np.asarray(rgb).copy()


def add_salt_pepper(image, amount, seed):
    noisy = image.copy()
    rng = np.random.default_rng(seed)
    mask = rng.random(image.shape[:2])
    noisy[mask < amount / 2] = 0
    noisy[mask > 1 - amount / 2] = 255
    return noisy


def save_image(image, path):
    Image.fromarray(image, mode="RGB").save(path)


def measure(function, image, kernel_size):
    started = time.perf_counter()
    result = function(image, kernel_size)
    elapsed = time.perf_counter() - started
    return result, elapsed


def draw_chart(rows, path):
    font = ImageFont.truetype(FONT_PATH, 16)
    small = ImageFont.truetype(FONT_PATH, 14)
    width, height = 780, 460
    margin_left, margin_right = 70, 210
    margin_top, margin_bottom = 56, 64
    chart = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(chart)

    max_time = max(row["native"] for row in rows) * 1.18
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    group_width = plot_width / len(rows)
    bar_width = group_width * 0.28
    baseline = height - margin_bottom

    draw.text((margin_left, 16), "время, с", fill=(20, 20, 20), font=font)
    draw.line(
        [(margin_left, margin_top), (margin_left, baseline)],
        fill=(40, 40, 40),
        width=2,
    )
    draw.line(
        [(margin_left, baseline), (width - margin_right, baseline)],
        fill=(40, 40, 40),
        width=2,
    )

    for row_index, row in enumerate(rows):
        origin = margin_left + group_width * row_index + group_width * 0.16
        native_height = plot_height * row["native"] / max_time
        pillow_height = max(plot_height * row["pillow"] / max_time, 2)
        draw.rectangle(
            [origin, baseline - native_height, origin + bar_width, baseline],
            fill=(36, 99, 168),
        )
        draw.rectangle(
            [
                origin + bar_width + 8,
                baseline - pillow_height,
                origin + 2 * bar_width + 8,
                baseline,
            ],
            fill=(214, 96, 48),
        )
        draw.text(
            (origin, baseline + 14),
            f"{row['kernel']}x{row['kernel']}",
            fill=(20, 20, 20),
            font=small,
        )
        draw.text(
            (origin, baseline - native_height - 20),
            f"{row['native']:.2f}",
            fill=(36, 99, 168),
            font=small,
        )

    legend_x = width - margin_right + 24
    draw.rectangle([legend_x, 80, legend_x + 18, 98], fill=(36, 99, 168))
    draw.text((legend_x + 26, 78), "нативный Python", fill=(20, 20, 20), font=small)
    draw.rectangle([legend_x, 112, legend_x + 18, 130], fill=(214, 96, 48))
    draw.text((legend_x + 26, 110), "Pillow", fill=(20, 20, 20), font=small)
    chart.save(path)


def run(args):
    results_dir = Path(args.output)
    results_dir.mkdir(parents=True, exist_ok=True)

    scene = load_image(IMAGE_PATH)
    noisy = add_salt_pepper(scene, args.noise, args.seed)
    save_image(scene, results_dir / "scene.png")
    save_image(noisy, results_dir / "noisy.png")

    rows = []
    for kernel_size in args.kernels:
        native, native_time = measure(median_filter_native, noisy, kernel_size)
        pillow, pillow_time = measure(median_filter_pillow, noisy, kernel_size)
        difference = int(np.max(np.abs(native.astype(np.int16) - pillow.astype(np.int16))))
        save_image(native, results_dir / f"native_k{kernel_size}.png")
        save_image(pillow, results_dir / f"pillow_k{kernel_size}.png")
        rows.append(
            {
                "kernel": kernel_size,
                "native": native_time,
                "pillow": pillow_time,
                "difference": difference,
            }
        )
        print(
            f"ядро {kernel_size}x{kernel_size}: "
            f"native {native_time:.3f} с, pillow {pillow_time:.3f} с, "
            f"|native-pillow| = {difference}"
        )

    draw_chart(rows, results_dir / "timings.png")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Медианный фильтр: нативный Python и Pillow"
    )
    parser.add_argument("--kernels", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--output", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
