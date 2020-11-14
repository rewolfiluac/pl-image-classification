from pathlib import Path

import cv2
from torch.utils.data import Dataset


EXTENTION = [
    ".tiff",
    ".jpg",
    ".png"
]


def cv2_loader(path):
    img = cv2.imread(path)
    return img


class ImageDataset(Dataset):
    def __init__(self, dataset_cfg, transform=None, loader=cv2_loader):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.transform = transform
        self.loader = loader
        self.classes = []
        self.data = []
        self.label = []
        self.prepare_data()

    def prepare_data(self):
        data_root = Path(self.dataset_cfg.root)
        self.classes = [
            p.stem for p in list(data_root.glob("*")) if p.is_dir()
        ]
        self.classes = sorted(self.classes)
        for cls_idx in range(len(self.classes)):
            cls_path = data_root / self.classes[cls_idx]
            _data = [p for p in list(cls_path.glob("*"))
                     if p.suffix in EXTENTION]
            self.data.extend(_data)
            self.label.extend(
                [cls_idx for _ in range(len(_data))]
            )

    def get_all_data(self):
        return self.data, self.label

    def __getitem__(self, idx):
        img_path, target = self.data[idx], self.label[idx]
        img = self.loader(str(img_path))
        if img is None:
            raise ValueError(
                "The image loaded by the dataset was None. Path: {}".format(
                    str(img_path))
            )
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, target

    def __len__(self):
        return len(self.data)
