import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pad, resize, to_tensor

from process_csv import readCsv


def data_path(n):
    return os.path.join(os.path.dirname(__file__), n)


class HWRSegmentationDataset(Dataset):
    """Dataset for all form images with annontated segments"""

    def __init__(self, csv_dir, form_dir, transform=None):
        """
        Args:
            csv_dir (string): Path to data/csv folder
            form_dir (string): Path to data/forms folder
            transform (callable?): Optional transform on data
        """
        self.csv_dir = csv_dir
        self.csv_list = os.listdir(csv_dir)
        self.form_dir = form_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_list)

    def __getitem__(self, idx):
        csv_filename = self.csv_list[idx]
        csv_path = os.path.join(self.csv_dir, csv_filename)
        boxes = readCsv(csv_path)

        img_filename = csv_filename.replace(".csv", ".png")
        img_path = os.path.join(self.form_dir, img_filename)
        img = Image.open(img_path).convert("L")

        result = {
            'image': img,
            'boxes': boxes
            # boxes are in [x, y, w, h] format, with (x, y) being the
            # top-left anchor and all units in absolute pixels
        }

        if self.transform:
            result = self.transform(result)

        return result


class MakeSquareAndTargets(object):
    def __init__(self, output_dim=512, grid_dim=32):
        self.output_dim = output_dim
        self.grid_dim = grid_dim

    def __call__(self, sample):
        image, boxes = sample['image'], sample['boxes']

        # IMAGE
        old_w, old_h = image.size[0], image.size[1]

        if old_h > old_w:
            new_h = self.output_dim
            new_w = int((self.output_dim * old_w / old_h) // 2 * 2)
            # making sure shorter dimension is even so padding works
            resize_ratio = new_h / old_h
            padding = (int((self.output_dim - new_w) / 2), 0)
        else:
            new_w = self.output_dim
            new_h = int(self.output_dim * old_h / old_w // 2 * 2)
            resize_ratio = new_w / old_w
            padding = (0, int((self.output_dim - new_w) / 2))

        # padding is (padding_h, padding_w)

        image = resize(image, (new_h, new_w))
        image = pad(image, padding)

        # LABEL
        # Transform x, y, w, h
        boxes = np.array(boxes)
        boxes[:, 0] = boxes[:, 0] * resize_ratio + padding[1]
        boxes[:, 1] = boxes[:, 1] * resize_ratio + padding[0]
        boxes[:, 2] = boxes[:, 2] * resize_ratio
        boxes[:, 3] = boxes[:, 3] * resize_ratio

        # convert to targets of [offset_x, offset_y, w, h, confidence]
        # target is [height x width]
        target = np.zeros((self.grid_dim, self.grid_dim, 5))
        grid_width = self.output_dim / self.grid_dim  # default is 32
        for box in boxes:
            # i, j are column, row of grid in target
            x, y, w, h = box[0], box[1], box[2], box[3]
            i = int(x // grid_width)
            j = int(y // grid_width)
            offset_x = (x - (i + 0.5) * grid_width) / grid_width
            offset_y = (y - (j + 0.5) * grid_width) / grid_width
            target_w = w / self.output_dim
            target_h = h / self.output_dim
            target[j][i] = [offset_x, offset_y, target_w, target_h, 1]
        target = np.moveaxis(target, 2, 0)
        return {
            "image": to_tensor(image),
            "target": target
        }
