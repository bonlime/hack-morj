"""Cut images and masks into overlapping patches. Overlap is needed to make training data more diverse"""

import json
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import Polygon
from tqdm import tqdm

from tiles import TileInference


def _poly_from_points(pts: np.array) -> Polygon:
    assert len(pts.shape) == 2 and pts.shape[0] == 1, "Shape is different from expected"
    pts = pts[0]
    if len(pts) < 6:
        print(f"Found poly with only {len(pts) // 2} points. Skipping.")
        return Polygon()
    # need buffer to fix invalid geometries. NOTE: not sure that it always fixes them correctly!
    return Polygon(zip(pts[::2], pts[1::2])).buffer(0)


def read_polys(annotation_path: Path) -> List[Polygon]:
    markup = json.load(annotation_path.open())
    assert all("segmentation_poly" in item for item in markup), "Some objects have no poly!"
    # taking zero element because they polys have shape [1, N]
    pts = [np.array(item["segmentation_poly"]) for item in markup if item["segmentation_poly"] is not None]
    polygons = [_poly_from_points(pt) for pt in pts]
    polygons = [p for p in polygons if p.area > 0]
    return polygons


BUFFER_CONSTANT = 1200  # very empirical constant to buffer masks


def _get_buffer_size(poly):
    "idea is to have small adaptivity in border size for different morjes"
    return max(min(int(round(poly.area / BUFFER_CONSTANT)), 5), 3)


def poly2mask(polygons: List[Polygon], out_shape: Tuple[int, int]) -> Tuple[np.array, np.array]:
    """Returns:
    mask for all morjes combined, mask for border between morjes
    """
    # binary mask for all morjes combined
    morj = features.rasterize(polygons, out_shape=out_shape, default_value=1)

    # slightly dilate polygons to make borders thicker
    # use adaptive buffer size based on morj area
    # get border by mask summation
    dilated_polys = [p.buffer(_get_buffer_size(p)) for p in polygons]
    morj_contact_raw = features.rasterize(dilated_polys, out_shape=out_shape, merge_alg=rasterio.enums.MergeAlg.add)
    morj_contact = (morj_contact_raw > 1).astype(np.uint8)

    # use morph closing for even better borders
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morj_contact = cv2.morphologyEx(morj_contact, cv2.MORPH_CLOSE, kernel)

    morj_border = (morj_contact_raw > 0) - morj
    morj_border = np.clip(morj_border + morj_contact, 0, 1)
    return np.stack([morj, morj_contact, morj_border], axis=2) * 255


IMG_PATCHES_DIR = "image_patches"
MASK_PATCHES_DIR = "mask_patches"


def main():
    # get args
    parser = ArgumentParser()
    parser.add_argument("--input_data_dir", type=Path, help="Folder with original data")
    parser.add_argument("--output_data_dir", type=Path, help="Folder where to put processed patches")
    parser.add_argument("--output_full_data_dir", type=Path, help="Optional path to store full masks")
    parser.add_argument("--tile_size", default=512, type=int)
    parser.add_argument("--tile_step", default=256, type=int)
    args = parser.parse_args()

    images = sorted((args.input_data_dir / "images").glob("*.jpg"))
    annotations = sorted((args.input_data_dir / "markup").glob("*.json"))
    # remove categories json, it's not used
    if (args.input_data_dir / "markup" / "categories.json") in annotations:
        annotations.pop(annotations.index(args.input_data_dir / "markup" / "categories.json"))
    assert len(images) == len(annotations), f"len(images) != len(annotations). {len(images)} != {len(annotations)}"
    print(f"Found {len(images)} images and annotations.")

    if args.output_data_dir is not None:
        shutil.rmtree(args.output_data_dir, ignore_errors=True)
        args.output_data_dir.mkdir(exist_ok=True)
        (args.output_data_dir / IMG_PATCHES_DIR).mkdir(exist_ok=True)
        (args.output_data_dir / MASK_PATCHES_DIR).mkdir(exist_ok=True)

    if args.output_full_data_dir is not None:
        shutil.rmtree(args.output_full_data_dir, ignore_errors=True)
        args.output_full_data_dir.mkdir(exist_ok=True)

    tiler = TileInference(tile_size=args.tile_size, tile_step=args.tile_step)
    for img, ann in tqdm(zip(images, annotations), total=len(images)):
        np_img = cv2.imread(str(img))
        polygons = read_polys(ann)
        mask = poly2mask(polygons, np_img.shape[:2])

        if args.output_full_data_dir is not None:
            mask_morj = mask[..., 0] * ((255 - mask[..., 2]) // 255)  # leave only centers of morj
            cv2.imwrite(str(args.output_full_data_dir / f"{img.stem}.png"), mask_morj)

        if args.output_data_dir is None:
            continue

        img_tiles = tiler.iter_split(np_img, channels_last=True)
        mask_tiles = tiler.iter_split(mask, channels_last=True)

        # use tile iterator to generate overlapping patches
        for idx, ((img_tile, _), (mask_tile, _)) in enumerate(zip(img_tiles, mask_tiles)):
            img_output_name = args.output_data_dir / IMG_PATCHES_DIR / f"{img.stem}_{idx}.jpg"
            mask_output_name = args.output_data_dir / MASK_PATCHES_DIR / f"{img.stem}_{idx}.png"
            cv2.imwrite(str(img_output_name), img_tile)
            cv2.imwrite(str(mask_output_name), mask_tile)

    print("Finished generating patches")


if __name__ == "__main__":
    main()
