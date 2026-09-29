"""Prepare the downsampled RGB/depth files required by Probe3D's NAVI reader.

Uses Probe3D's resize recipe: EXIF-transpose, resize the short side to 1024,
bicubic RGB and nearest-neighbor depth, and prefix each output name with
``downsampled_``. Originals are retained.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps


def prepare_one(path):
    path = Path(path)
    destination = path.with_name("downsampled_" + path.name)
    if destination.is_file():
        return False
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        width, height = image.size
        scale = 1024 / min(width, height)
        size = (int(width * scale), int(height * scale))
        method = Image.Resampling.BICUBIC if path.suffix.lower() == ".jpg" else Image.Resampling.NEAREST
        resized = image.resize(size, method)
        temporary = destination.with_name(destination.name + f".{os.getpid()}.part")
        try:
            resized.save(temporary, format="JPEG" if path.suffix.lower() == ".jpg" else "PNG")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Extracted navi_v1 directory")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if not args.root.is_dir() or args.workers < 1:
        parser.error("root must exist and workers must be positive")
    paths = sorted(path for scene in args.root.glob("*/*") for folder, suffix in
                   (("images", ".jpg"), ("depth", ".png"))
                   for path in (scene / folder).glob(f"*{suffix}")
                   if not path.name.startswith("downsampled_"))
    if not paths:
        raise ValueError(f"No NAVI source images/depth files under {args.root}")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        converted = sum(pool.map(prepare_one, paths))
    print(f"NAVI resize ready: {converted} new, {len(paths) - converted} existing, {len(paths)} source files", flush=True)


if __name__ == "__main__":
    main()
