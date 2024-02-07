from typing import Any, Dict, Optional, Tuple, List

from scipy.ndimage import affine_transform
import scipy.ndimage
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import glob

import os
import numpy as np
import cv2

from tqdm import tqdm

stats = {
    "kidney_2": {"xmin": 29784.0, "xmax": 42380.0},
    "kidney_2_rot": {"xmin": 29784.0, "xmax": 42380.0},
    "kidney_3_sparse_xz": {"xmin": 18966.0, "xmax": 21944.0},
    "kidney_1_dense_zy": {"xmin": 20243.0, "xmax": 29649.0},
    "kidney_1_voi_xz": {"xmin": 29646.0, "xmax": 45297.0},
    "kidney_2_xz": {"xmin": 29784.0, "xmax": 42380.0},
    "kidney_1_dense": {"xmin": 20243.0, "xmax": 29649.0},
    "kidney_1_dense_rot": {"xmin": 20243.0, "xmax": 29649.0},
    "kidney_3_dense": {"xmin": 18806.0, "xmax": 21903.0},
    "kidney_2_zy": {"xmin": 29784.0, "xmax": 42380.0},
    "kidney_3_sparse_zy": {"xmin": 18966.0, "xmax": 21944.0},
    "kidney_1_voi": {"xmin": 29646.0, "xmax": 45297.0},
    "kidney_3_sparse": {"xmin": 18966.0, "xmax": 21944.0},
    "kidney_3_sparse_rot": {"xmin": 18966.0, "xmax": 21944.0},
    "kidney_1_voi_zy": {"xmin": 29646.0, "xmax": 45297.0},
    "kidney_1_dense_xz": {"xmin": 20243.0, "xmax": 29649.0},
    "kidney_3_dense_zy": {"xmin": 18806.0, "xmax": 21903.0},
    "kidney_3_dense_xz": {"xmin": 18806.0, "xmax": 21903.0},
    "50.16um_LADAF_2020-27_kidney-left_jp2_": {"xmin": 19138.0, "xmax": 21940.0},
    "50.16um_LADAF_2020-27_kidney-left_jp2__xz": {"xmin": 19138.0, "xmax": 21940.0},
    "50.16um_LADAF_2020-27_kidney-left_jp2__zy": {"xmin": 19138.0, "xmax": 21940.0},
    "50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2__xz": {
        "xmin": 15500.0,
        "xmax": 35605.0,
    },
    "50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2__zy": {
        "xmin": 15500.0,
        "xmax": 35605.0,
    },
    "50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2_": {"xmin": 15500.0, "xmax": 35605.0},
}

folds = {
    0: {
        "train": [
            "kidney_1_dense",
            "kidney_2",
            "50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2_",
        ],
        "val": [
            "kidney_3_dense",
        ],
    },
    1: {
        "train": [
            "kidney_1_dense",
            # "kidney_3_dense",
            "kidney_3_sparse",
            "50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2_",
        ],
        "val": ["kidney_2"],
    },
    2: {
        "train": [
            "kidney_2",
            "kidney_3_dense",
            "kidney_3_sparse",
        ],
        "val": [
            "kidney_1_dense",
        ],
    },
    -1: {
        "train": [
            "kidney_1_dense",
            "kidney_2",
            "kidney_1_voi",
            "kidney_3_dense",
            "kidney_3_sparse",
            "50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2_",
        ],
        "val": [
            # "kidney_3_sparse",
            "kidney_2",
            # "kidney_1_dense"
        ],
    },
    "debug": {
        "train": [
            "kidney_3_dense",
        ],
        "val": [
            "kidney_3_dense",
        ],
    },
}

spacing = {
    "kidney_1_dense": 50,
    "kidney_1_voi": 5.2,
    "kidney_2": 50,
    "kidney_3_dense": 50.16,
    "kidney_3_sparse": 50.16,
    "kidney_5": 50.28,
    "kidney_6": 63.08,
}


class VesselDataset3D(torch.utils.data.Dataset):
    def __init__(
        self,
        fold: int = 0,
        mode: str = "train",
        data_dir: str = "/home/igor/working/",
        transforms: Any = None,
        crop_shape: Tuple = (256, 256, 256),
        samples_per_epoch: int = 10000,
    ):
        self.ids = folds[fold][mode]
        self.mode = mode
        self.dataset_path = os.path.join(data_dir, "blood-vessel-segmentation", "train")
        self.transforms = transforms
        self.crop_shape = crop_shape
        self.samples_per_epoch = samples_per_epoch
        self.data = {}
        self.init_dataset()
        print(f"Dataset size: {len(self.ids)}")

    def init_dataset(self):
        self.data = {
            _id: self.load_volume(os.path.join(self.dataset_path, _id, "labels", "*"))
            for _id in self.ids
        }

    def __len__(self):
        return self.samples_per_epoch

    def normilize(self, image: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
        image = (image - xmin) / (xmax - xmin)
        image = np.clip(image, 0, 1)
        return image.astype(np.float32)

    @staticmethod
    def norm_by_percentile(
        volume: np.ndarray, low: float = 10, high: float = 99.8
    ) -> Tuple:
        xmin = np.percentile(volume, low)
        xmax = np.max([np.percentile(volume, high), 1])
        return xmin, xmax

    def load_volume(
        self,
        path: str,
    ) -> Dict:
        dataset = sorted(glob.glob(path))

        volume = None
        target = None

        for z, path in enumerate(tqdm(dataset)):
            mask = (cv2.imread(path, 0) > 127.0) * 1.0
            path = path.replace(
                "labels",
                "images",
            ).replace(".png", ".jp2")
            if "/kidney_3_dense/" in path:
                path = path.replace("kidney_3_dense", "kidney_3_sparse")

            image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
            image = np.array(image, dtype=np.uint16)
            if volume is None:
                volume = np.zeros((len(dataset), *image.shape[-2:]), dtype=np.uint16)
                target = np.zeros((len(dataset), *mask.shape[-2:]), dtype=np.uint8)
            volume[z] = image
            target[z] = mask

        return {"volume": volume, "target": target}

    def random_affine_transform(self, volume, mask):
        # Define random translation, rotation, and scaling parameters
        translation = np.random.uniform(-10, 10, 3)  # Adjust range as needed
        scaling_factor = np.random.uniform(0.8, 1.2, 3)  # Scaling factor

        # Create affine transformation matrix
        transform_matrix = np.eye(4)

        # Apply rotation
        if np.random.rand() > 0.5:
            rotation_axes = [(0, 1), (0, 2), (1, 2)]
            axis = np.random.choice([0, 1, 2])

            rotation_angle = np.random.uniform(-45, 45)  # Rotation angle in degrees
            volume = scipy.ndimage.rotate(
                volume,
                rotation_angle,
                axes=rotation_axes[axis],
                reshape=False,
                order=1,
                mode="mirror",
            )
            mask = scipy.ndimage.rotate(
                mask,
                rotation_angle,
                axes=rotation_axes[axis],
                reshape=False,
                order=0,
                mode="mirror",
            )

        # Apply translation and scaling
        for i in range(3):
            transform_matrix[i, -1] = translation[i]
            transform_matrix[i, i] = scaling_factor[i]

        # Apply affine transformation
        volume = affine_transform(
            volume, matrix=transform_matrix, order=1, mode="mirror"
        )
        mask = affine_transform(mask, matrix=transform_matrix, order=0, mode="mirror")

        return volume, mask

    def random_augmentation(self, volume, mask):
        # Random rotation (90-degree increments)
        rotation_axes = [(0, 1), (0, 2), (1, 2)]
        axis = np.random.choice([0, 1, 2])
        angle = np.random.choice([0, 90, 180, 270])
        volume = np.rot90(volume, angle // 90, axes=rotation_axes[axis])
        mask = np.rot90(mask, angle // 90, axes=rotation_axes[axis])

        # Random flips
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=0)
            mask = np.flip(mask, axis=0)
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=1)
            mask = np.flip(mask, axis=1)
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=2)
            mask = np.flip(mask, axis=2)

        # if np.random.rand() > 0.5:
        #     original_shape = volume.shape
        #
        #     zoom_factor = 1 + np.random.uniform(-0.2, 0.2)
        #     volume = zoom(
        #         volume, zoom_factor, order=1
        #     )  # Using bilinear interpolation for volume
        #     mask = zoom(
        #         mask, zoom_factor, order=0
        #     )  # Using nearest-neighbor interpolation for mask
        #
        #     # If zooming out, crop to original size. If zooming in, pad with zeros to original size.
        #     volume, mask = self.match_shape(volume, original_shape), self.match_shape(
        #         mask, original_shape
        #     )

        # if np.random.rand() > 0.3:
        #     brightness_factor = np.random.uniform(
        #         0.8, 1.1
        #     )  # Adjust this range as needed
        #     volume = (volume * brightness_factor).astype(np.uint16)
        #     volume = np.clip(volume, 0, 65535)

        return volume, mask

    @staticmethod
    def match_shape(tensor, target_shape, padding="reflect"):
        """
        Match the shape of a tensor by padding or center cropping with reflective padding.

        Parameters:
        - tensor: Input tensor
        - target_shape: Target shape (tuple)

        Returns:
        - Tensor with the matched shape
        """
        if tensor.shape == target_shape:
            return tensor  # No need for adjustment if shapes are already equal

        output_tensor = np.zeros(target_shape, dtype=tensor.dtype)

        # Calculate start and end indices for the tensor
        start_indices = [
            max(0, (s - t) // 2) for s, t in zip(tensor.shape, target_shape)
        ]
        end_indices = [
            start + min(s, t)
            for start, s, t in zip(start_indices, tensor.shape, target_shape)
        ]

        # Calculate start and end indices for the output tensor
        output_start = [
            max(0, (t - s) // 2) for s, t in zip(tensor.shape, target_shape)
        ]
        output_end = [
            start + min(s, t)
            for start, s, t in zip(output_start, tensor.shape, target_shape)
        ]

        # Calculate padding values
        pad_before = [start if start > 0 else 0 for start in output_start]
        pad_after = [
            t - end if end < t else 0 for end, t in zip(output_end, target_shape)
        ]

        # Reflective padding
        tensor_padded = np.pad(
            tensor,
            (
                (pad_before[0], pad_after[0]),
                (pad_before[1], pad_after[1]),
                (pad_before[2], pad_after[2]),
            ),
            mode=padding,
        )

        # Copy the padded tensor into the output tensor
        output_tensor[
            output_start[0] : output_end[0],
            output_start[1] : output_end[1],
            output_start[2] : output_end[2],
        ] = tensor_padded[
            start_indices[0] : end_indices[0],
            start_indices[1] : end_indices[1],
            start_indices[2] : end_indices[2],
        ]

        return output_tensor

    def __getitem__(self, item):
        # if self.mode == "train":
        random_id = np.random.choice(self.ids)
        data = self.data[random_id]

        volume = data["volume"]
        target = data["target"]
        xmin = stats[random_id]["xmin"]
        xmax = stats[random_id]["xmax"]

        if self.mode == "train":
            sample_non_empty_mask = bool(np.random.binomial(n=1, p=0.5))

            sample_new_mask = True
            # Random Crop
            while sample_new_mask:
                start_x = np.random.randint(0, volume.shape[0] - self.crop_shape[0])
                start_y = np.random.randint(0, volume.shape[1] - self.crop_shape[1])
                start_z = np.random.randint(0, volume.shape[2] - self.crop_shape[2])

                volume_crop = volume[
                    start_x : start_x + self.crop_shape[0],
                    start_y : start_y + self.crop_shape[1],
                    start_z : start_z + self.crop_shape[2],
                ].copy()

                target_crop = target[
                    start_x : start_x + self.crop_shape[0],
                    start_y : start_y + self.crop_shape[1],
                    start_z : start_z + self.crop_shape[2],
                ].copy()

                sample_new_mask = sample_non_empty_mask and target_crop.sum() == 0

                volume_crop, target_crop = self.random_augmentation(
                    volume_crop.copy(), target_crop.copy()
                )
        else:
            start_x = np.random.randint(0, volume.shape[0] - self.crop_shape[0])
            start_y = np.random.randint(0, volume.shape[1] - self.crop_shape[1])
            start_z = np.random.randint(0, volume.shape[2] - self.crop_shape[2])

            volume_crop = volume[
                start_x : start_x + self.crop_shape[0],
                start_y : start_y + self.crop_shape[1],
                start_z : start_z + self.crop_shape[2],
            ].copy()

            target_crop = target[
                start_x : start_x + self.crop_shape[0],
                start_y : start_y + self.crop_shape[1],
                start_z : start_z + self.crop_shape[2],
            ].copy()

        volume_crop = self.normilize(volume_crop, xmin=xmin, xmax=xmax)

        volume_crop, target_crop = np.ascontiguousarray(
            volume_crop
        ), np.ascontiguousarray(target_crop)

        return {
            "volume": np.expand_dims(volume_crop, axis=0),
            "target": np.expand_dims(target_crop, axis=0),
            "id": random_id,
        }


class DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/home/igor/working",
        fold: int = 0,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        img_size: List = (256, 256, 256),
        samples_per_epoch: int = 250,
        val_samples_per_epoch: int = 50,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # #TODO: add augs
        self.transforms = {"train": [], "val": []}

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        # self.train_dataset_ft: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """

    def setup(self, stage: Optional[str] = None) -> None:
        # TODO: move dataloading here

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
        self.valid_dataset = VesselDataset3D(
            fold=self.hparams.fold,
            mode="val",
            data_dir=self.hparams.data_dir,
            transforms=self.transforms["val"],
            crop_shape=self.hparams.img_size,
            samples_per_epoch=self.hparams.val_samples_per_epoch,
        )

        self.train_dataset = VesselDataset3D(
            fold=self.hparams.fold,
            mode="train",
            data_dir=self.hparams.data_dir,
            transforms=self.transforms["train"],
            crop_shape=self.hparams.img_size,
            samples_per_epoch=self.hparams.samples_per_epoch,
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

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size_per_device,  # 1
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
            dataset=self.valid_dataset,
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
    dataset = VesselDataset3D(mode="train", fold="debug")
    for i in range(len(dataset)):
        data = dataset[i]
        print(data["volume"].shape)
        # plt.subplot(121)
        # plt.imshow(np.mean(data["volume"].squeeze(), axis=0))
        # plt.subplot(122)
        # plt.imshow(np.mean(data["target"].squeeze(), axis=0))
        # plt.show()

    # datamodule = DataModule()
    # datamodule.prepare_data()
    # datamodule.setup()
    #
    # for i in range(100):
    #     data = datamodule.train_dataset[i]
    #     print(data["volume"].shape, data["id"])
    #     # img, msk = data["image"], data["mask"]
    #     # img = (img.permute((1, 2, 0)) * 255).numpy().astype("uint8")
    #     # msk = (msk.permute((1, 2, 0)) * 255).numpy().astype("uint8")
    #     # plt.figure(figsize=(9, 4))
    #     # print(data_name)
    #     # plt.axis("off")
    #     # plt.subplot(1, 3, 1)
    #     # plt.imshow(img)
    #     # plt.subplot(1, 3, 2)
    #     # plt.imshow(msk)
    #     # plt.subplot(1, 3, 3)
    #     # plt.imshow(img, cmap="bone")
    #     # plt.imshow(msk, alpha=0.5)
    #     # plt.show()
    # print("")
    #
    # for i in range(2):
    #     data = datamodule.valid_dataset[i]
    #     print(data["volume"].shape, data["id"])
    # print("")
