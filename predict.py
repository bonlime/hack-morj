import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

import albumentations as A
import cv2
import segmentation_models_pytorch as smp
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from tqdm import tqdm

from tiles import TileInference


class MorjInferenceDataset(Dataset):
    def __init__(self, data_dir, augmentation):
        self.images = sorted(data_dir.glob("*.jpg"))
        self.augmentation = augmentation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.images[idx]))[:, :, ::-1].copy()
        return self.augmentation(image=image)["image"]


def get_test_aug():
    aug = A.Compose(
        [
            A.Normalize(),
            ToTensorV2(transpose_mask=True),
        ]
    )
    return aug


def predict():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=Path, help="Folder with train data")
    parser.add_argument("--data_dir", type=Path, help="Folder with val data")
    parser.add_argument("--output_dir", type=Path, help="Folder with val data")
    parser.add_argument(
        "--tiler_kwargs",
        type=json.loads,
        default='{"tile_size": 512, "tile_step": 256, "tile_pad": 0, "fusion": "pyramid"}',
    )
    args = parser.parse_args()

    shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(exist_ok=True)
    ckpt_name = next((args.checkpoint_dir / "checkpoints").glob("*.ckpt"))
    chpt = torch.load(ckpt_name, map_location="cpu")
    # remove `model.` prefix for weights
    state_dict = {k[6:]: v for k, v in chpt["state_dict"].items()}

    hparams = yaml.load((args.checkpoint_dir / "hparams.yaml").open(), yaml.SafeLoader)
    model = smp.create_model(**hparams["model_kwargs"])
    model.load_state_dict(state_dict)
    model = model.cuda().eval().requires_grad_(False)

    dataset = MorjInferenceDataset(args.data_dir, get_test_aug())
    tiler = TileInference(model=model, **args.tiler_kwargs)

    for single_img, single_image_name in tqdm(zip(dataset, dataset.images), total=len(dataset)):
        output_name = args.output_dir / single_image_name.with_suffix(".png").name
        print(f"Predicting: {output_name}")
        input_img = single_img[None].cuda()
        pred = tiler(input_img).sigmoid()[0].permute(1, 2, 0)
        pred = torch.cat([pred, torch.zeros(*pred.shape[:2], 1)], 2)
        pred = (pred * 255).to(torch.uint8).numpy()
        cv2.imwrite(str(output_name), pred)


if __name__ == "__main__":
    predict()
