import json
from argparse import ArgumentParser
from pathlib import Path

import albumentations as A
import cv2
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.callbacks import StochasticWeightAveraging
from torch.utils.data import DataLoader, Dataset

from prepare_data import IMG_PATCHES_DIR, MASK_PATCHES_DIR


class MorjDataset(Dataset):
    def __init__(self, data_dir, augmentation):
        self.images = sorted((data_dir / IMG_PATCHES_DIR).glob("*.jpg"))
        self.masks = sorted((data_dir / MASK_PATCHES_DIR).glob("*.png"))
        self.augmentation = augmentation
        assert len(self.images) == len(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.images[idx]))[:, :, ::-1].copy()
        mask = cv2.imread(str(self.masks[idx]))
        aug = self.augmentation(image=image, mask=mask)
        image = aug["image"].float()
        mask = aug["mask"].float() / 255
        return image, mask


def get_train_aug(size):
    aug = A.Compose(
        [
            A.OneOrOther(
                A.RandomResizedCrop(size, size, (0.15, 0.4), interpolation=cv2.INTER_CUBIC),
                A.RandomCrop(size, size),
            ),
            A.Flip(),
            A.RandomRotate90(),
            A.SomeOf(
                [
                    A.OneOrOther(
                        A.GaussNoise(var_limit=(2, 20)), A.GaussNoise(var_limit=(2, 20), per_channel=False), p=0.3
                    ),
                    A.OneOf([A.Blur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=(3, 5)), A.GaussianBlur()], p=0.3),
                    A.RandomContrast(p=0.3),
                    A.HueSaturationValue(p=0.3),
                    A.RandomBrightness(p=0.3),
                    A.RandomGamma(p=0.3),
                    A.CLAHE(p=0.3),
                ],
                n=7,  # Select all transforms in random order.
                p=1,  # Always apply
                replace=False,  # Don't repeat same transform
            ),
            A.Normalize(),
            ToTensorV2(transpose_mask=True),
        ]
    )
    return aug


def get_val_aug(size):
    aug = A.Compose(
        [
            A.CenterCrop(size, size),
            A.Normalize(),
            ToTensorV2(transpose_mask=True),
        ]
    )
    return aug


class MorjModel(pl.LightningModule):
    BORDER_SIZE = 24
    CONTACT_WEIGHT = 3
    BORDER_WEIGHT = 1

    def __init__(self, model_kwargs):
        super().__init__()
        self.model = smp.create_model(**model_kwargs)
        self.save_hyperparameters("model_kwargs")
        self.border_mask = None

    def forward(self, image):
        return self.model(image)

    def shared_step(self, batch):
        self.lr_schedulers().step()
        images, masks = batch
        logits_mask = self.forward(images)

        # more weights for contact because it's more important
        loss_mask = F.binary_cross_entropy_with_logits(
            logits_mask[:, 2],
            masks[:, 2],
            pos_weight=torch.tensor(2),
            reduction="none",
        )
        loss_contact = self.CONTACT_WEIGHT * F.binary_cross_entropy_with_logits(
            logits_mask[:, 1],
            masks[:, 1],
            pos_weight=torch.tensor(2),
            reduction="none",
        )
        loss_border = self.BORDER_WEIGHT * F.binary_cross_entropy_with_logits(
            logits_mask[:, 0],
            masks[:, 0],
            pos_weight=torch.tensor(2),
            reduction="none",
        )

        loss = loss_mask + loss_contact + loss_border

        if self.border_mask is None:
            # most errors are on boundary, so ignore them
            side = torch.ones(images.size(2))
            side[: self.BORDER_SIZE] = torch.linspace(0, 1, self.BORDER_SIZE)
            side[-self.BORDER_SIZE :] = torch.linspace(1, 0, self.BORDER_SIZE)
            self.border_mask = (
                (side[:, None] @ side[None]).view(1, 1, images.size(2), images.size(2)).to(images).detach()
            )

        loss = (loss * self.border_mask).mean()

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0002, weight_decay=0.001)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=ARGS.epochs)
        return [optimizer], [scheduler]
        # return optimizer


ARGS = None


def set_args():
    parser = ArgumentParser()
    parser.add_argument("--train_data_dir", type=Path, help="Folder with train data")
    parser.add_argument("--val_data_dir", type=Path, help="Folder with val data")
    parser.add_argument("--model_kwargs", type=json.loads, default='{"arch": "Unet", "encoder_name": "resnet34"}')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=256)
    global ARGS
    ARGS = parser.parse_args()


def train():
    set_args()
    model_kwargs = dict(in_channels=3, classes=3, **ARGS.model_kwargs)
    model = MorjModel(model_kwargs=model_kwargs)

    train_dataloader = DataLoader(
        MorjDataset(ARGS.train_data_dir, get_train_aug(ARGS.patch_size)),
        batch_size=ARGS.batch_size,
        num_workers=12,
        drop_last=True,
        shuffle=True,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        MorjDataset(ARGS.val_data_dir, get_val_aug(ARGS.patch_size)),
        batch_size=ARGS.batch_size,
        num_workers=12,
        drop_last=True,
        shuffle=False,
        pin_memory=True,
    )

    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        max_epochs=ARGS.epochs,
        check_val_every_n_epoch=5,
        callbacks=[StochasticWeightAveraging(swa_epoch_start=0.5)],
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )


if __name__ == "__main__":
    train()
