"""Build the README gallery from the unchanged original PNG files.

Run from any directory with Python, Pillow and Matplotlib installed.
"""

from pathlib import Path

from matplotlib import get_data_path
from PIL import Image, ImageChops, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PANELS = (
    ("THB_fun_ref_side.png", "THB basis: three levels\nof refinement", (800, 740)),
    ("scordelis.png", "Trimmed roof with\na circular opening", (800, 740)),
    ("Gengar.png", "Trimmed and locally refined\nspline geometry", (690, 690)),
)


def crop_white_margin(image):
    """Remove outer white space, retaining a small safety margin."""
    difference = ImageChops.difference(image, Image.new("RGB", image.size, "white"))
    mask = difference.point(lambda value: 255 if value > 25 else 0)
    bounds = mask.getbbox()
    if bounds is None:
        raise ValueError("The source image has no visible content.")
    left, top, right, bottom = bounds
    padding = 8
    return image.crop((max(0, left - padding), max(0, top - padding),
                       min(image.width, right + padding),
                       min(image.height, bottom + padding)))


def main():
    # Three identical 860 x 800 image areas, with 30 px gutters and outer margins.
    gallery = Image.new("RGB", (2700, 1000), "white")
    draw = ImageDraw.Draw(gallery)
    font_path = Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    font = ImageFont.truetype(str(font_path), 42)

    for index, (filename, caption, image_size) in enumerate(PANELS):
        with Image.open(ROOT / "pictures" / filename) as source:
            image = crop_white_margin(source.convert("RGB"))

        # Fit proportionally; preserve the roof's legend and the surrounding mesh.
        image.thumbnail(image_size, Image.Resampling.LANCZOS)
        center_x = 460 + index * 890
        gallery.paste(image, (center_x - image.width // 2,
                              30 + (800 - image.height) // 2))

        # Fixed baselines keep both caption lines aligned across all panels.
        for line_index, line in enumerate(caption.splitlines()):
            draw.text((center_x, 860 + line_index * 56), line, font=font,
                      fill="#24292f", anchor="mt")

    output = ROOT / "pictures/readme_gallery.png"
    gallery.save(output, optimize=True)
    print(f"Created {output.name}: {gallery.width} x {gallery.height} px")


if __name__ == "__main__":
    main()
