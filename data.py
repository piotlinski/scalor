import os.path
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import random
from common import *
import re
from PIL import Image, ImageFile
from pathlib import Path

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TrainStation(Dataset):
    def __init__(self, args, train=False):

        self.args = args
        root = os.path.expanduser(self.args.data_dir)
        self.phase_train = train
        if self.phase_train:
            self.path = os.path.join(root, "train")
        else:
            self.path = os.path.join(root, "test")

        self.sequences = list(sorted(Path(self.path).glob("*")))
        self.anns = [json.load(p.joinpath("annotations.json").open()) for p in self.sequences]


    @staticmethod
    def load_image(path):
        return np.array(Image.open(path))

    def __getitem__(self, index):
        sequence = self.sequences[index]
        ann = self.anns[index]

        images_files = list(sorted(sequence.glob("*.jpg")))
        imgs = np.stack(self.load_image(p) for p in images_files).transpose(0, 3, 1, 2)
        imgs = imgs.astype(np.float) / 255.0
        imgs = torch.from_numpy(imgs).float()

        return imgs, torch.zeros(1)

    def __len__(self):
        return len(self.sequences)
