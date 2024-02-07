from typing import Any, Dict, Optional, List
import cv2
import numpy as np
import albumentations as A
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os

# from src.data.components.transforms import ChannelDropout

stats = {
    "kidney_2": {"xmin": 29784.0, "xmax": 42380.0, "weight": 0.65},
    "kidney_2_rot": {"xmin": 29784.0, "xmax": 42380.0, "weight": 0.65},
    "kidney_3_sparse_xz": {"xmin": 18966.0, "xmax": 21944.0, "weight": 0.85},
    "kidney_1_dense_zy": {"xmin": 20243.0, "xmax": 29649.0, "weight": 1.0},
    "kidney_1_voi_xz": {"xmin": 29646.0, "xmax": 45297.0, "weight": 1.0},
    "kidney_2_xz": {"xmin": 29784.0, "xmax": 42380.0, "weight": 0.65},
    "kidney_1_dense": {"xmin": 20243.0, "xmax": 29649.0, "weight": 1.0},
    "kidney_1_dense_rot": {"xmin": 20243.0, "xmax": 29649.0, "weight": 1.0},
    "kidney_3_dense": {"xmin": 18806.0, "xmax": 21903.0, "weight": 1.0},
    "kidney_2_zy": {"xmin": 29784.0, "xmax": 42380.0, "weight": 0.65},
    "kidney_3_sparse_zy": {"xmin": 18966.0, "xmax": 21944.0, "weight": 0.85},
    "kidney_1_voi": {"xmin": 29646.0, "xmax": 45297.0, "weight": 1.0},
    "kidney_3_sparse": {"xmin": 18966.0, "xmax": 21944.0, "weight": 0.85},
    "kidney_3_sparse_rot": {"xmin": 18966.0, "xmax": 21944.0, "weight": 0.85},
    "kidney_1_voi_zy": {"xmin": 29646.0, "xmax": 45297.0, "weight": 1.0},
    "kidney_1_dense_xz": {"xmin": 20243.0, "xmax": 29649.0, "weight": 1.0},
    "kidney_3_dense_zy": {"xmin": 18806.0, "xmax": 21903.0, "weight": 1.0},
    "kidney_3_dense_xz": {"xmin": 18806.0, "xmax": 21903.0, "weight": 0.85},
    "50.16um_LADAF_2020-27_kidney-left_jp2_": {
        "xmin": 19138.0,
        "xmax": 21940.0,
        "weight": 0.65,
    },
    "50.16um_LADAF_2020-27_kidney-left_jp2__xz": {
        "xmin": 19138.0,
        "xmax": 21940.0,
        "weight": 0.65,
    },
    "50.16um_LADAF_2020-27_kidney-left_jp2__zy": {
        "xmin": 19138.0,
        "xmax": 21940.0,
        "weight": 0.65,
    },
    "50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2__xz": {
        "xmin": 15500.0,
        "xmax": 35605.0,
        "weight": 0.65,
    },
    "50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2__zy": {
        "xmin": 15500.0,
        "xmax": 35605.0,
        "weight": 0.65,
    },
    "50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2_": {
        "xmin": 15500.0,
        "xmax": 35605.0,
        "weight": 0.65,
    },
}

folds = {
    0: {
        "train_groups": [
            "kidney_1_dense",
            "kidney_2",
            "kidney_1_dense_rot",
            "kidney_2_rot",
            # "kidney_1_voi",
            # "kidney_1_voi_xy",
            # "kidney_1_voi_zy",
            "kidney_1_dense_xz",
            "kidney_1_dense_zy",
            # "kidney_1_voi_xz",
            # "kidney_1_voi_zy",
            "kidney_2_xz",
            "kidney_2_zy",
            "50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2_",
            "50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2__xz",
            "50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2__zy",
        ],
        "valid_groups": [
            "kidney_3_dense",  # "kidney_3_dense_xz", kidney_3_dense_zy
            # "kidney_3_sparse"
        ],
    },
    1: {
        "train_groups": [
            "kidney_3_sparse_rot",
            "kidney_1_dense_rot",
            "kidney_1_dense",
            "kidney_3_dense",
            "kidney_3_dense_xz",
            "kidney_3_dense_zy",
            # "kidney_1_voi",
            # "kidney_1_voi_xy",
            # "kidney_1_voi_zy",
            "kidney_1_dense_xz",
            "kidney_1_dense_zy",
            # "kidney_1_voi_xz",
            # "kidney_1_voi_zy",
            "kidney_3_sparse",
            "kidney_3_sparse_zy",
            "kidney_3_sparse_xz",
            "50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2_",
            "50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2__xz",
            "50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2__zy",
        ],
        "valid_groups": ["kidney_2"],
    },
    2: {
        "train_groups": [
            "kidney_3_sparse_rot",
            "kidney_2_rot",
            "kidney_2",
            "kidney_2_xz",
            "kidney_2_zy",
            "kidney_3_dense",
            "kidney_3_dense_xz",
            "kidney_3_dense_zy",
            # "kidney_1_voi",
            # "kidney_1_voi_xz",
            # "kidney_1_voi_zy",
            "kidney_3_sparse",
            "kidney_3_sparse_zy",
            "kidney_3_sparse_xz",
        ],
        "valid_groups": [
            "kidney_1_dense",
        ],
    },
    -1: {
        "train_groups": [
            "kidney_3_sparse_rot",
            "kidney_1_dense_rot",
            "kidney_2_rot",
            "kidney_1_dense",
            "kidney_2",
            # "kidney_1_voi",
            # "kidney_1_voi_xy",
            # "kidney_1_voi_zy",
            "kidney_3_dense",
            "kidney_3_dense_xz",
            "kidney_3_dense_zy",
            "kidney_3_sparse",
            "kidney_3_sparse_zy",
            "kidney_3_sparse_xz",
            "kidney_1_dense_xz",
            "kidney_1_dense_zy",
            # "kidney_1_voi_xz",
            # "kidney_1_voi_zy",
            "kidney_2_xz",
            "kidney_2_zy",
            "50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2_",
            "50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2__xz",
            "50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2__zy",
        ],
        "valid_groups": [
            "kidney_3_dense",
        ],
    },
}

folds_ft = {
    0: {
        "train_groups": [
            "kidney_1_dense",
            "kidney_1_dense_xz",
            "kidney_1_dense_zy",
        ],
        "valid_groups": [
            "kidney_3_dense",
        ],
    },
    1: {
        "train_groups": [
            "kidney_1_dense",
            "kidney_3_dense",
            "kidney_3_dense_xz",
            "kidney_3_dense_zy",
            "kidney_1_dense_xz",
            "kidney_1_dense_zy",
        ],
        "valid_groups": ["kidney_2"],
    },
    2: {
        "train_groups": [
            "kidney_3_dense",
            "kidney_3_dense_xz",
            "kidney_3_dense_zy",
        ],
        "valid_groups": [
            "kidney_1_dense",
        ],
    },
}


class VesselDatasetV2(torch.utils.data.Dataset):
    def __init__(
        self,
        img_paths: List,
        msk_paths: List = [],
        transforms: Any = None,
        in_channels: int = 3,
        train_mutliplier: int = 1,
        norm_scheme: str = "max",
        upscale: bool = False,
        train_pseudo: bool = False,
        mode: str = "train",
    ):
        self.img_paths = sorted(img_paths)
        self.msk_paths = sorted(msk_paths)
        self.transforms = transforms
        assert in_channels % 2 == 1
        self.in_channels = in_channels
        self.train_mutliplier = train_mutliplier
        self.norm_scheme = norm_scheme
        self.upscale = upscale
        self.train_pseudo = train_pseudo
        self.mode = mode

        self.img_dict = {}

        for path_i in self.img_paths:
            parts = path_i.split("/")
            key = parts[-3]
            if key not in self.img_dict:
                self.img_dict[key] = []
            self.img_dict[key].append(path_i)

        class_weights = np.array(
            [stats[key]["weight"] for key in sorted(list(self.img_dict.keys()))]
        )
        self.class_weights = class_weights / class_weights.sum()

    def __len__(self):
        return len(self.img_paths) * self.train_mutliplier

    def load_img(self, path, folder):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if self.upscale:
            img = cv2.resize(img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        img = np.tile(img[..., None], [1, 1, 1])  # gray to rgb
        img = img.astype("float32")  # original is uint16
        mx = np.max(img)
        mn = np.min(img)
        if self.norm_scheme == "max":
            img /= mx  # scale image to [0, 1]
        elif self.norm_scheme == "minmax":
            # if mx:
            img -= mn
            img /= mx  # scale image to [0, 1]
        else:
            img = (img - stats[folder]["xmin"]) / (
                stats[folder]["xmax"] - stats[folder]["xmin"]
            )
            img = np.clip(img, 0, 1)
        return img

    def load_msk(self, path):
        # mask = cv2.imread(path, cv2.IMREAD_UNCHANGED) / 255.0
        if self.train_pseudo:
            if "kidney" in path:
                mask = (
                    cv2.imread(
                        path.replace("/train/", "/train_pseudo/"), cv2.IMREAD_UNCHANGED
                    )
                    > 127
                ).astype(np.float32)
        else:
            mask = (cv2.imread(path, cv2.IMREAD_UNCHANGED) > 127).astype(np.float32)

        if self.upscale:
            mask = cv2.resize(mask, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        return mask

    def find_neighboring_files(self, file_path: str, limit: int = 1) -> List:
        """
        Find neighboring files with consecutive numbers in the filename.

        Args:
        - file_path (str): The path of the original file.
        - limit (int or None): The maximum number of neighboring files on each side to include. If None, include all neighbors.

        Returns:
        - list: A list of neighboring files with full paths.
        """
        limit = int(limit)
        img = self.load_img(file_path, file_path.split("/")[-3])

        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)

        # Extract the base name and extension
        base_name, extension = os.path.splitext(filename)

        # Find the prefix and suffix of the filename
        prefix = base_name.rstrip("0123456789")
        suffix = base_name[len(prefix) :]

        # Find all files in the directory with the same prefix and extension
        neighboring_files: List = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.startswith(prefix) and f.endswith(extension)
        ]

        # Sort the files based on the numeric part of the filename
        neighboring_files.sort(
            key=lambda x: int(
                x[len(directory) + len(os.path.sep) + len(prefix) : -len(suffix)]
            )
            if suffix
            else int(x[len(directory) + len(os.path.sep) + len(prefix) :])
        )

        # Find the index of the input file in the sorted list
        index = neighboring_files.index(file_path)

        # Calculate the start and end indices for the desired neighbors
        start_index: int = int(max(0, index - limit) if limit is not None else 0)
        end_index: int = int(
            min(len(neighboring_files), index + limit + 1)
            if limit is not None
            else len(neighboring_files)
        )

        # Limit the number of neighbors
        if index - limit < 0:
            neighboring_files = [np.zeros_like(img)] * (
                limit - index
            ) + neighboring_files[start_index:end_index]
        elif index + limit + 1 > len(neighboring_files):
            neighboring_files = neighboring_files[start_index:end_index] + [
                np.zeros_like(img)
            ] * (1 + limit - (len(neighboring_files) - index))
        else:
            neighboring_files = neighboring_files[start_index:end_index]

        return neighboring_files

    def load_img25d(self, img_path, folder, limit):
        return np.concatenate(
            [
                self.load_img(f, folder) if isinstance(f, str) else f
                for f in self.find_neighboring_files(img_path, limit)
            ],
            -1,
        )

    def compute_contours_mask(self, mask, thickness=3):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(contours_mask, contours, -1, 1, thickness=thickness)
        return contours_mask

    def compute_center_of_mass_mask(self, mask, radius=3):
        labeled_mask = mask.astype(np.uint8)
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(labeled_mask)

        center_of_mass_mask = np.zeros_like(mask, dtype=np.uint8)
        for label in range(1, num_labels):  # Exclude background label (0)
            center = tuple(map(int, centroids[label]))
            cv2.circle(center_of_mass_mask, center, radius, 1, thickness=-1)

        return center_of_mass_mask

    def compute_distance_transform(self, mask):
        mask = (mask * 255).astype(np.uint8)
        distance_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        # Normalize distance transform to [0, 1]
        distance_transform = distance_transform / (distance_transform.max() + 1)
        return distance_transform

    def __getitem__cutmix__(self, folder):
        img_path = np.random.choice(self.img_dict[folder])

        parts = img_path.split("/")

        image_id = f"{parts[-3]}_{parts[-1].split('.')[0]}"

        if self.in_channels > 1:
            img = self.load_img25d(img_path, parts[-3], (self.in_channels - 1) / 2)
        else:
            img = self.load_img(img_path, parts[-3])

        orig_size = img.shape

        # print(img.shape)
        if len(self.msk_paths) > 0:
            # msk = self.load_msk(msk_path)
            msk = self.load_msk(
                img_path.replace("images", "labels").replace(".jp2", ".png")
            )
        else:
            msk = np.zeros_like(img)

        # try:
        if self.transforms:
            data = self.transforms(image=img, mask=msk)
            img = data["image"]
            msk = data["mask"]

        # contours = self.compute_contours_mask(msk)
        # center_of_mass = self.compute_center_of_mass_mask(msk)
        # dtm = self.compute_distance_transform(msk)

        img = np.transpose(img, (2, 0, 1))
        return {
            "image": torch.tensor(img),
            "mask": torch.tensor(np.expand_dims(msk, 0)),
            # "contours": torch.tensor(np.expand_dims(contours, 0)),
            # "center_of_mass": torch.tensor(np.expand_dims(center_of_mass, 0)),
            "shape": torch.tensor(np.array([orig_size[0], orig_size[1]])),
            # "dtm": torch.tensor(np.expand_dims(dtm, 0)),
            "id": image_id,
            "folder": parts[-3],
        }

    def __getitem__(self, index):
        # TODO: find bug here!!!
        # TODO: make better sampling like for 3d
        if self.mode != "train":
            index = index % len(self.img_paths)

            img_path = self.img_paths[index]
        else:
            # if np.random.random() < :
            # folder =
            folder = np.random.choice(
                sorted(list(self.img_dict.keys())), p=self.class_weights
            )
            img_path = np.random.choice(self.img_dict[folder])

        parts = img_path.split("/")

        image_id = f"{parts[-3]}_{parts[-1].split('.')[0]}"

        if self.in_channels > 1:
            img = self.load_img25d(img_path, parts[-3], (self.in_channels - 1) / 2)
        else:
            img = self.load_img(img_path, parts[-3])

        orig_size = img.shape

        # print(img.shape)
        if len(self.msk_paths) > 0:
            # msk = self.load_msk(msk_path)
            msk = self.load_msk(
                img_path.replace("images", "labels").replace(".jp2", ".png")
            )
        else:
            msk = np.zeros_like(img)

        # try:
        if self.transforms:
            data = self.transforms(image=img, mask=msk)
            img = data["image"]
            msk = data["mask"]

        # contours = self.compute_contours_mask(msk)
        # center_of_mass = self.compute_center_of_mass_mask(msk)
        # dtm = self.compute_distance_transform(msk)

        img = np.transpose(img, (2, 0, 1))
        return {
            "image": torch.tensor(img),
            "mask": torch.tensor(np.expand_dims(msk, 0)),
            # "contours": torch.tensor(np.expand_dims(contours, 0)),
            # "center_of_mass": torch.tensor(np.expand_dims(center_of_mass, 0)),
            "shape": torch.tensor(np.array([orig_size[0], orig_size[1]])),
            # "dtm": torch.tensor(np.expand_dims(dtm, 0)),
            "id": image_id,
            "folder": parts[-3],
        }
        # except ValueError:
        #     print(img_path, msk_path)
        #     print(msk.shape, img.shape)


class CutMixDataset(Dataset):
    def __init__(self, dataset, num_mix=1, beta=1.0, prob=1.0, transforms=None):
        self.dataset = dataset
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob
        self.transforms = transforms

    @staticmethod
    def rand_bbox(size, lam):
        if len(size) == 4:
            w = size[2]
            h = size[3]
        elif len(size) == 3:
            w = size[1]
            h = size[2]
        else:
            raise Exception

        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int32(w * cut_rat)
        cut_h = np.int32(h * cut_rat)

        # uniform
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def __getitem__(self, index):
        data = self.dataset[index]
        img, target = data["image"], data["mask"]
        if np.random.random() < 0.5:
            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)

            data2 = self.dataset.__getitem__cutmix__(data["folder"])

            img2, target2 = data2["image"], data2["mask"]

            bbx1, bby1, bbx2, bby2 = self.rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            target[:, bbx1:bbx2, bby1:bby2] = target2[:, bbx1:bbx2, bby1:bby2]

        if self.transforms:
            data_aug = self.transforms(
                image=img.numpy().transpose(1, 2, 0),
                mask=target.numpy().transpose(1, 2, 0),
            )
            img = torch.tensor(np.transpose(data_aug["image"], (2, 0, 1)))
            target = torch.tensor(np.transpose(data_aug["mask"], (2, 0, 1)))

        return {
            "image": img,
            "mask": target,
            "shape": data["shape"],
            "id": data["id"],
        }

    def __len__(self):
        return len(self.dataset)


class DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/home/igor/working/",
        gt_df: str = "/home/igor/working/blood-vessel-segmentation/gt_extended.csv",
        fold: int = 0,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        img_size: List = [512, 512],
        aug_scheme: str = "d4",
        use_cutmix: bool = False,
        in_channels: int = 3,
        train_mutliplier: int = 1,
        norm_scheme: str = "max",
        upscale: bool = False,
        train_pseudo: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.transforms = {
            "train": self.init_transforms(img_size),
            "valid": A.Compose(
                [
                    # A.CenterCrop(*img_size)
                    # A.RandomCrop(*img_size, p=1.0),
                    # A.Resize(*img_size, interpolation=cv2.INTER_NEAREST),
                ],
                p=1.0,
            ),
        }

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        # self.train_dataset_ft: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

        train_groups = folds[self.hparams.fold]["train_groups"]
        # train_groups_ft = folds_ft[self.hparams.fold]["train_groups"]
        valid_groups = folds[self.hparams.fold]["valid_groups"]
        gt_df = pd.read_csv(self.hparams.gt_df)
        gt_df["img_path"] = gt_df["img_path"].apply(
            lambda x: os.path.join(self.hparams.data_dir, x)
        )
        gt_df["msk_path"] = gt_df["msk_path"].apply(
            lambda x: os.path.join(self.hparams.data_dir, x)
        )
        self.train_df = gt_df.query("group in @train_groups").reset_index(drop=True)
        # self.train_df_ft = gt_df.query("group in @train_groups_ft").reset_index(
        #     drop=True
        # )
        self.valid_df = gt_df.query("group in @valid_groups").reset_index(drop=True)

    def init_transforms(self, crop_size):
        if self.hparams.aug_scheme == "hard":
            return A.Compose(
                [
                    # A.Resize(*img_size, interpolation=cv2.INTER_NEAREST),
                    A.PadIfNeeded(*crop_size),
                    A.CropNonEmptyMaskIfExists(*crop_size, p=1.0),
                    A.ShiftScaleRotate(
                        # border_mode=cv2.BORDER_CONSTANT,
                        # value=0,
                        # mask_value=0,
                        scale_limit=0.2,
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(),
                            A.RandomBrightness(),
                            A.RandomGamma(),
                        ],
                        p=1.0,
                    ),
                ],
                p=1.0,
            )
        elif self.hparams.aug_scheme == "d4":
            return A.Compose(
                [
                    A.PadIfNeeded(*crop_size),
                    A.CropNonEmptyMaskIfExists(*crop_size, p=1.0),
                    # A.RandomCrop(*crop_size, p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ],
                p=1.0,
            )
        elif self.hparams.aug_scheme == "kernel":
            return A.Compose(
                [
                    A.Rotate(limit=270, p=0.25),
                    A.RandomScale(
                        scale_limit=(0.8, 1.25), interpolation=cv2.INTER_CUBIC, p=0.25
                    ),
                    A.RandomCrop(*crop_size, p=1),
                    A.RandomBrightnessContrast(p=0.25),
                    A.GaussianBlur(p=0.1),
                    A.MotionBlur(p=0.1),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.1),
                ]
            )

        elif self.hparams.aug_scheme == "severe":
            return A.Compose(
                [
                    # A.Resize(*img_size, interpolation=cv2.INTER_NEAREST),
                    A.PadIfNeeded(*crop_size),
                    A.CropNonEmptyMaskIfExists(*crop_size, p=1.0),
                    A.ShiftScaleRotate(
                        # border_mode=cv2.BORDER_CONSTANT,
                        # value=0,
                        # mask_value=0,
                        scale_limit=0.2,
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(),
                            A.RandomBrightness(),
                            A.RandomGamma(),
                        ],
                        p=1.0,
                    ),
                    # ChannelDropout(),
                    A.Cutout(),
                ],
                p=1.0,
            )
        elif self.hparams.aug_scheme == "severe_v2":
            return A.Compose(
                [
                    # A.Resize(*img_size, interpolation=cv2.INTER_NEAREST),
                    A.PadIfNeeded(*crop_size),
                    A.CropNonEmptyMaskIfExists(*crop_size, p=1.0),
                    A.OneOf(
                        [
                            # cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4
                            A.ShiftScaleRotate(
                                # border_mode=cv2.BORDER_CONSTANT,
                                # value=0,
                                # mask_value=0,
                                scale_limit=0.2,
                                interpolation=cv2.INTER_NEAREST,
                            ),
                            A.ShiftScaleRotate(
                                # border_mode=cv2.BORDER_CONSTANT,
                                # value=0,
                                # mask_value=0,
                                scale_limit=0.2,
                                interpolation=cv2.INTER_LINEAR,
                            ),
                            # A.ShiftScaleRotate(
                            #     # border_mode=cv2.BORDER_CONSTANT,
                            #     # value=0,
                            #     # mask_value=0,
                            #     scale_limit=0.2,
                            #     interpolation=cv2.INTER_AREA,
                            # ),
                        ]
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(),
                            A.RandomBrightness(),
                            A.RandomGamma(),
                        ],
                        p=1.0,
                    ),
                    # ChannelDropout(),
                    A.Cutout(),
                ],
                p=1.0,
            )

        elif self.hparams.aug_scheme == "d4_scaled":
            return A.Compose(
                [
                    A.PadIfNeeded(*crop_size),
                    # A.CropNonEmptyMaskIfExists(*crop_size, p=1.0),
                    A.CropNonEmptyMaskIfExists(*crop_size, p=1.0),
                    A.ShiftScaleRotate(
                        # border_mode=cv2.BORDER_CONSTANT,
                        # value=0,
                        # mask_value=0,
                        rotate_limit=0,
                        scale_limit=0.2,
                    ),
                    # A.RandomCrop(*crop_size, p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ],
                p=1.0,
            )

        elif self.hparams.aug_scheme == "hard_random":
            return A.Compose(
                [
                    A.PadIfNeeded(*crop_size),
                    # A.Resize(*img_size, interpolation=cv2.INTER_NEAREST),
                    A.RandomCrop(*crop_size, p=1.0),
                    A.ShiftScaleRotate(
                        scale_limit=0.2,
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(),
                            A.RandomGamma(),
                        ],
                        p=1.0,
                    ),
                ],
                p=1.0,
            )

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.

        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # load and split datasets only if not loaded already
        train_img_paths = self.train_df["img_path"].values.tolist()
        train_msk_paths = self.train_df["msk_path"].values.tolist()

        # train_img_paths_ft = self.train_df_ft["img_path"].values.tolist()
        # train_msk_paths_ft = self.train_df_ft["msk_path"].values.tolist()

        valid_img_paths = self.valid_df["img_path"].values.tolist()
        valid_msk_paths = self.valid_df["msk_path"].values.tolist()

        # self.train_dataset_ft = VesselDataset(
        #     train_img_paths_ft,
        #     train_msk_paths_ft,
        #     transforms=self.transforms["train"],
        #     train_mutliplier=self.hparams.train_mutliplier,
        #     norm_scheme=self.hparams.norm_scheme,
        #     in_channels=self.hparams.in_channels,
        #     upscale=self.hparams.upscale,
        #     train_pseudo=self.hparams.train_pseudo
        # )

        if self.hparams.use_cutmix:
            train_dataset = VesselDatasetV2(
                train_img_paths,
                train_msk_paths,
                transforms=None,
                train_mutliplier=self.hparams.train_mutliplier,
                norm_scheme=self.hparams.norm_scheme,
                in_channels=self.hparams.in_channels,
                upscale=self.hparams.upscale,
                train_pseudo=self.hparams.train_pseudo,
                mode="train",
            )
            self.train_dataset = CutMixDataset(
                dataset=train_dataset, transforms=self.transforms["train"]
            )
        else:
            self.train_dataset = VesselDatasetV2(
                train_img_paths,
                train_msk_paths,
                transforms=self.transforms["train"],
                train_mutliplier=self.hparams.train_mutliplier,
                norm_scheme=self.hparams.norm_scheme,
                in_channels=self.hparams.in_channels,
                upscale=self.hparams.upscale,
                train_pseudo=self.hparams.train_pseudo,
                mode="train",
            )

        self.valid_dataset = VesselDatasetV2(
            valid_img_paths,
            valid_msk_paths,
            transforms=self.transforms["valid"],
            norm_scheme=self.hparams.norm_scheme,
            in_channels=self.hparams.in_channels,
            upscale=self.hparams.upscale,
            mode="valid",
        )

        self.test_dataset = VesselDatasetV2(
            valid_img_paths,
            [],
            transforms=self.transforms["valid"],
            norm_scheme=self.hparams.norm_scheme,
            in_channels=self.hparams.in_channels,
            upscale=self.hparams.upscale,
            mode="test",
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=False,
        )
        # if self.trainer.current_epoch <= 15:
        #     return DataLoader(
        #         dataset=self.train_dataset,
        #         batch_size=self.batch_size_per_device,
        #         num_workers=self.hparams.num_workers,
        #         pin_memory=self.hparams.pin_memory,
        #         shuffle=True,
        #     )
        # else:
        #     return DataLoader(
        #         dataset=self.train_dataset_ft,
        #         batch_size=self.batch_size_per_device,
        #         num_workers=self.hparams.num_workers,
        #         pin_memory=self.hparams.pin_memory,
        #         shuffle=True,
        #     )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=1,  # self.batch_size_per_device // 2,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=1,  # self.batch_size_per_device // 2,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    datamodule = DataModule(
        aug_scheme="hard",
        use_cutmix=True,
        norm_scheme="minmax_organ",
        in_channels=1,
        train_mutliplier=2,
    )

    datamodule.prepare_data()
    datamodule.setup()

    # sample_ids = [random.randint(0, len(datamodule.train_df)) for _ in range(5)]
    # sample_ids = range(5)
    sample_ids = range(len(datamodule.train_df))
    # sample_ids = range(920, 925)
    for sample_id in sample_ids:
        data_name = datamodule.train_df.loc[sample_id]["id"]
        data = datamodule.train_dataset[sample_id]
        print(data["image"].shape, data["id"])
        # img, msk = data["image"], data["mask"]
        # img = (img.permute((1, 2, 0)) * 255).numpy().astype("uint8")
        # msk = (msk.permute((1, 2, 0)) * 255).numpy().astype("uint8")
        # plt.figure(figsize=(9, 4))
        # print(data_name)
        # plt.axis("off")
        # plt.subplot(1, 3, 1)
        # plt.imshow(img)
        # plt.subplot(1, 3, 2)
        # plt.imshow(msk)
        # plt.subplot(1, 3, 3)
        # plt.imshow(img, cmap="bone")
        # plt.imshow(msk, alpha=0.5)
        # plt.show()
    print("")
