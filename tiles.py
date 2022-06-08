"""
Implementation of tile-based inference allowing to predict huge images that does not fit into GPU memory entirely
in a sliding-window fashion and merging prediction mask back to full-resolution.

Reference:
    https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/inference/tiles.py
    https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
    https://github.com/victorca25/iNNfer
"""
import itertools
import math
from typing import Callable, Optional

import numpy as np
import torch
from torch.nn import functional as F


def compute_pyramid_patch_weight_loss(width: int, height: int) -> torch.Tensor:
    """Compute a weight matrix that assigns bigger weight on pixels in center and
    less weight to pixels on image boundary. This weight matrix then used for merging
    individual tile predictions and helps dealing with prediction artifacts on tile boundaries.

    Args:
        width: Tile width
        height: Tile height

    Returns:
        Weight matrix of shape (1, width, height)

    """
    xc = width * 0.5
    yc = height * 0.5
    xl = 0
    xr = width
    yb = 0
    yt = height
    Dc = np.zeros((width, height))
    De = np.zeros((width, height))

    Dcx = np.square(np.arange(width) - xc + 0.5)
    Dcy = np.square(np.arange(height) - yc + 0.5)
    Dc = np.sqrt(Dcx[np.newaxis].transpose() + Dcy)

    De_l = np.square(np.arange(width) - xl + 0.5) + np.square(0.5)
    De_r = np.square(np.arange(width) - xr + 0.5) + np.square(0.5)
    De_b = np.square(0.5) + np.square(np.arange(height) - yb + 0.5)
    De_t = np.square(0.5) + np.square(np.arange(height) - yt + 0.5)

    De_x = np.sqrt(np.minimum(De_l, De_r))
    De_y = np.sqrt(np.minimum(De_b, De_t))
    De = np.minimum(De_x[np.newaxis].transpose(), De_y)

    alpha = (width * height) / np.sum(np.divide(De, np.add(Dc, De)))
    W = alpha * np.divide(De, np.add(Dc, De))
    return torch.tensor(W)[None, ...]


class TileInference:
    """Wrapper for models that implements tile-based inference allowing to predict huge images
    that does not fit into GPU memory entirely in a sliding-window fashion.

    Args:
        model: Any callable used for processing single input patch.
        fusion: One of {'mean', 'pyramid'}. Defines how overlapped patches are weighted.
    """

    def __init__(
        self,
        tile_size: int = 512,
        tile_step: int = 256,
        tile_pad: int = 0,
        scale: int = 1,
        model: Optional[Callable] = None,
        fusion: str = "mean",
    ):
        # Check input values
        assert tile_size >= tile_step, "Tile step can't be larger than tile size"

        self.tile_size = tile_size
        self.tile_step = tile_step
        self.tile_pad = tile_pad

        # By how many pixels tiles are overlapped with each other
        self.overlap = tile_size - tile_step

        # self.pre_pad = pre_pad
        self.scale = scale
        self.model = model

        weights = {"mean": self._mean, "pyramid": self._pyramid}
        self.weight = weights[fusion](tile_size * scale)

    def _mean(self, tile_size):
        return torch.ones((1, tile_size, tile_size))

    def _pyramid(self, tile_size):
        return compute_pyramid_patch_weight_loss(tile_size, tile_size)

    def __call__(self, image: torch.Tensor, out_channels: Optional[int] = None):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Args:
            image: 4D tensor with shape (N, C, H, W)

        """
        # Save initial image size
        B, C, H, W = image.shape

        # Number of tiles in both directions.
        tiles_w = max(1, math.ceil((W - self.overlap) / self.tile_step))
        tiles_h = max(1, math.ceil((H - self.overlap) / self.tile_step))

        extra_w = self.tile_step * tiles_w - (W - self.overlap)
        extra_h = self.tile_step * tiles_h - (H - self.overlap)

        margin_left = 0
        margin_right = extra_w
        margin_top = 0
        margin_bottom = extra_h

        # Make image divisible by `tile_size` and add border pixels if necessary
        padded_image = F.pad(
            image,
            (
                margin_left + self.tile_pad,
                margin_right + self.tile_pad,
                margin_top + self.tile_pad,
                margin_bottom + self.tile_pad,
            ),
            "replicate",  # Other border types produce artifacts
        )

        tile_generator = self.iter_split(padded_image)

        # Output shape includes margin borders. We crop them out later
        output_shape = (
            B,
            C if out_channels is None else out_channels,
            (H + margin_bottom + margin_top) * self.scale,
            (W + margin_right + margin_left) * self.scale,
        )

        # Empty output tensor
        output = image.new_zeros(output_shape).cpu()
        # Used to save weight mask used for blending
        norm_mask = image.new_zeros(output_shape).cpu()

        # Move weights to correct device and dtype
        # w = self.weight.to(image.device)
        w = self.weight.to("cpu")

        for tile, (x, y, tile_width, tile_height) in tile_generator:
            with torch.no_grad():
                # Process
                output_tile = self.model(tile).cpu()

            # Remove added border pixels
            if self.tile_pad:
                output_tile = output_tile[
                    ...,
                    self.tile_pad * self.scale : -self.tile_pad * self.scale,
                    self.tile_pad * self.scale : -self.tile_pad * self.scale,
                ]
                tile_width -= 2 * self.tile_pad
                tile_height -= 2 * self.tile_pad

            output_x_left = x * self.scale
            output_x_right = (x + tile_width) * self.scale
            output_y_top = y * self.scale
            output_y_bottom = (y + tile_height) * self.scale

            output[..., output_y_top:output_y_bottom, output_x_left:output_x_right] += output_tile * w

            norm_mask[..., output_y_top:output_y_bottom, output_x_left:output_x_right] += w

        # Normalize by mask to weighten overlapped patches.
        output = torch.div(output, norm_mask)

        # Crop added margins
        output = output[
            ...,
            margin_top * self.scale : (H + margin_top) * self.scale,
            margin_left * self.scale : (W + margin_left) * self.scale,
        ]

        return output

    def iter_split(self, image, channels_last: bool = False):
        """

        Splits the image into partially overlapping patches.
        The patches overlap by padding_size pixels.
        # Pads the image twice:
        #     - first to have a size multiple of the patch size,
        #     - then to have equal padding at the borders.
        # Args:
        #     image_array: numpy array of the input image.
        #     patch_size: size of the patches from the original image (without padding).
        #     padding_size: size of the overlapping area.
        """
        if channels_last:
            image = image.swapaxes(0, -1)[None]  # HxWxC -> 1xCxWxH

        x_lefts = range(self.tile_pad, image.shape[3] - self.tile_size - self.tile_pad + 1, self.tile_step)
        y_tops = range(self.tile_pad, image.shape[2] - self.tile_size - self.tile_pad + 1, self.tile_step)

        # Loop over all tiles
        for x, y in itertools.product(x_lefts, y_tops):
            x_left = x - self.tile_pad
            x_right = x_left + self.tile_size + 2 * self.tile_pad
            y_top = y - self.tile_pad
            y_bottom = y_top + self.tile_size + 2 * self.tile_pad

            crop = (
                x - self.tile_pad,  # left coordinate
                y - self.tile_pad,  # top coordinate
                self.tile_size + 2 * self.tile_pad,  # crop size
                self.tile_size + 2 * self.tile_pad,  # crop size
            )

            tile = image[
                ...,
                y_top:y_bottom,  # height
                x_left:x_right,  # width
            ]

            if channels_last:
                tile = tile[0].swapaxes(0, -1)  # 1xCxWxH -> HxWxC

            yield tile, crop
