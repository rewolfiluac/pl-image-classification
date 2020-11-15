from pathlib import Path

import cv2
import torch
from tqdm import tqdm
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score)
from pytorch_lightning.trainer import seed_everything

from pl_module import LightningModuleInference
from utils.factory import get_transform
from utils.util import get_parser, read_yaml


TARGET_EXTENSION = [
    ".jpg",
    ".png",
    ".tiff"
]

TEST_ROOT_PATH = "../data/CIFAR10/test"
OUTPUT_PATH = "../data/CIFAR10/output"
CHECKPOINT = "/home/eseshinpu/projects/pl-image-classification/src/mlruns/0/21ba0a2721864310a23e7260e86b9cbf/artifacts/epoch=43-val_loss_mean=0.607-val_acc=0.818.pth.ckpt"


def inference(cfg):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    # define model
    pl_module = LightningModuleInference.load_from_checkpoint(
        CHECKPOINT, cfg=cfg
    ).eval().to(device)

    # define transform
    transform = get_transform(cfg.transform.val)

    test_root_path = Path(TEST_ROOT_PATH)
    output_path = Path(OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)

    # search test images
    img_paths = [
        p for p in list(test_root_path.glob("*/*"))
        if p.suffix in TARGET_EXTENSION
    ]

    labels, predicts = [], []
    for img_path in tqdm(img_paths):
        cls_name = img_path.parent.name
        org_img = cv2.imread(str(img_path))
        if org_img is None:
            raise Exception("NotFound Image: {}".format(str(img_path)))
        img = transform(image=org_img)["image"]
        input = img.unsqueeze(0).to(device, non_blocking=True)
        outputs = pl_module(input)
        _, predicted_indexes = torch.max(outputs.data, 1)
        predict_idx = predicted_indexes.cpu().numpy()
        labels.append(str(cls_name))
        predicts.append(str(predict_idx[0]))
    calc_eval(labels, predicts)


def calc_eval(labels, predicts):
    matrix = confusion_matrix(labels, predicts)
    acc = accuracy_score(labels, predicts)
    precision = precision_score(labels, predicts, average="weighted")
    recall = recall_score(labels, predicts, average="weighted")
    f1 = f1_score(labels, predicts, average="weighted")
    print(matrix)
    print("acc      :", acc)
    print("precision:", precision)
    print("recall   :", recall)
    print("f1       :", f1)


if __name__ == "__main__":
    args = get_parser().parse_args()

    cfg = read_yaml(path=args.config)

    seed_everything(seed=cfg.general.seed)
    inference(cfg)
