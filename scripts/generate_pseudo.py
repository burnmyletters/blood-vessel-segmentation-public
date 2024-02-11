import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from typing import Union, Dict, Tuple

import monai
import ttach as tta
import re
import argparse


def rename_keys(original_dict: Dict, pattern: str) -> Dict:
    new_dict = {}

    for old_key, value in original_dict.items():
        new_key = re.sub(pattern, "", old_key)

        new_dict[new_key] = value

    return new_dict


def to_device(
    x: Union[torch.nn.Module, torch.Tensor], cuda_id: int = 0
) -> Union[torch.nn.Module, torch.Tensor]:
    return x.cuda(cuda_id) if torch.cuda.is_available() else x


def load_jit_model(model_path: str, cuda_id: int = 0) -> torch.nn.Module:
    model = torch.jit.load(
        model_path,
        map_location=f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu",
    )
    return model


predict_on = [
    [
        "UnetPlusPlus",
        "tu-tf_efficientnet_b5",
        3,
        "UnetPlusPlus_tu-tf_efficientnet_b5_size_3_512_bs32_hard_minmax_organ_novoi_fx",
        1.0,
    ],
    [
        "UnetPlusPlus",
        "tu-tf_efficientnet_b6",
        3,
        "UnetPlusPlus_tu-tf_efficientnet_b6_src.models.components.losses.BoundaryDoULoss_size_3_512_bs32_hard_minmax_organ_novoi_fx",
        1.5,
    ],
]


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: str, in_channels: int = 3, is_test: bool = False):
        self.window = in_channels // 2
        self.is_test = is_test
        self.ids = []

        self.xmin, self.xmax = 0, 0

        self.data_tensor = self.load_volume(dataset)
        self.shape_orig = self.data_tensor.shape

        padding = ((self.window, self.window),) * self.data_tensor.ndim

        self.padding = tuple(
            (max(0, before), max(0, after)) for (before, after) in padding
        )
        self.data_tensor = np.pad(
            self.data_tensor, padding, mode="constant", constant_values=0
        )

    def __len__(self):
        return sum(self.shape_orig) if self.is_test else self.shape_orig[0]

    def normilize(self, image: np.ndarray) -> np.ndarray:
        image = (image - self.xmin) / (self.xmax - self.xmin)
        image = np.clip(image, 0, 1)
        return image.astype(np.float32)

    @staticmethod
    def norm_by_percentile(
        volume: np.ndarray, low: float = 10, high: float = 99.8
    ) -> Tuple:
        xmin = np.percentile(volume, low)
        print(xmin)
        xmax = np.max([np.percentile(volume, high), 1])
        print(xmax)
        return xmin, xmax

    def load_volume(self, dataset: str) -> np.ndarray:
        path = os.path.join(dataset, "images", "*.jp2")
        dataset = sorted(glob(path))
        for p_img in tqdm(dataset):
            path_ = p_img.split(os.path.sep)
            slice_id, _ = os.path.splitext(path_[-1])
            self.ids.append(f"{path_[-3]}_{slice_id}")

        volume = None

        for z, path in enumerate(tqdm(dataset)):
            image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
            image = np.array(image, dtype=np.uint16)
            if volume is None:
                volume = np.zeros((len(dataset), *image.shape[-2:]), dtype=np.uint16)
            volume[z] = image
        self.xmin, self.xmax = self.norm_by_percentile(volume)
        return volume

    def __getitem__(self, idx: int) -> Dict:
        # Determine which axis to sample from based on the index
        if idx < self.shape_orig[0]:
            idx = idx + self.window
            slice_data = self.normilize(
                self.data_tensor[
                    idx - self.window : 1 + idx + self.window, :, :
                ].transpose(1, 2, 0)[
                    self.window : -self.window, self.window : -self.window, :
                ]
            )
            axis = "X"
            idx -= 1

        elif idx < self.shape_orig[0] + self.shape_orig[1]:
            idx -= self.shape_orig[0] - self.window
            slice_data = self.normilize(
                self.data_tensor[
                    :, idx - self.window : 1 + idx + self.window, :
                ].transpose(0, 2, 1)[
                    self.window : -self.window, self.window : -self.window, :
                ]
            )
            axis = "Y"
            idx -= 1

        else:
            idx -= self.shape_orig[0] + self.shape_orig[1] - self.window

            slice_data = self.normilize(
                self.data_tensor[:, :, idx - self.window : 1 + idx + self.window][
                    self.window : -self.window, self.window : -self.window, :
                ]
            )
            axis = "Z"
            idx -= 1

        slice_data = torch.tensor(slice_data.transpose(2, 0, 1))

        return {"slice": slice_data, "slice_index": idx, "axis": axis}


def main():
    datasets = sorted(glob(f"{DATASET_FOLDER}/50um*_"))

    print(len(datasets))

    tta_models = []
    weights = []

    use_top_only = True
    folds2predict = [0, 1]

    for model_config in tqdm(predict_on):
        #     for fold in range(3):
        for fold in folds2predict:
            if use_top_only:
                model_path = sorted(
                    glob(
                        f"{logs_base_path}/{model_config[3]}/{fold}/checkpoints/epoch*.ckpt"
                    )
                )[-1]
                state_dict = rename_keys(
                    torch.load(model_path, map_location="cpu")["state_dict"], "net."
                )
                model = to_device(
                    smp.create_model(
                        arch=model_config[0],
                        encoder_name=model_config[1],
                        in_channels=model_config[2],
                        encoder_weights=None,
                    )
                )
                model.load_state_dict(state_dict)
                model.eval()
                # model = torch.nn.DataParallel(model)

                tta_models.append(model)

                tta_models.append(
                    tta.SegmentationTTAWrapper(
                        model, tta.aliases.d4_transform(), merge_mode="mean"
                    )
                )
                weights.append(model_config[-1])
            else:
                for model_path in sorted(
                    glob(
                        f"{logs_base_path}/{model_config[3]}/{fold}/checkpoints/epoch*.ckpt"
                    )
                ):
                    state_dict = rename_keys(
                        torch.load(model_path, map_location="cpu")["state_dict"], "net."
                    )
                    model = to_device(
                        smp.create_model(
                            arch=model_config[0],
                            encoder_name=model_config[1],
                            in_channels=model_config[2],
                            encoder_weights=None,
                        )
                    )

                    model.load_state_dict(state_dict)
                    model.eval()
                    # model = torch.nn.DataParallel(model)

                    tta_models.append(
                        tta.SegmentationTTAWrapper(
                            model, tta.aliases.d4_transform(), merge_mode="mean"
                        )
                    )
                    # tta_models.append(model)
                    weights.append(model_config[-1])

    os.makedirs(DATASET_FOLDER, exist_ok=True)

    rles, ids = [], []
    with torch.no_grad():
        for dataset in datasets:
            folder = dataset.split("/")[-1]
            os.makedirs(
                f"{DATASET_FOLDER}/{folder}/labels",
                exist_ok=True,
            )

            test_dataset = BuildDataset(dataset, is_test=True)
            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                num_workers=0,
                shuffle=False,
                pin_memory=False,
            )
            ls_images = sorted(glob(f"{dataset}/images/*.jp2"))

            print(f"processing {dataset}")
            print(f"found images: {len(ls_images)}")

            y_preds = np.zeros(test_dataset.shape_orig, dtype=np.half)

            pbar = tqdm(
                enumerate(test_loader),
                total=len(test_loader),
                desc=f"Inference {dataset}",
            )
            for step, batch in pbar:
                images = to_device(batch["slice"])
                axis = batch["axis"][0]
                idx = batch["slice_index"].numpy()[0]

                preds = 0
                for tta_model, weight in zip(tta_models, weights):
                    preds += weight * monai.inferers.sliding_window_inference(
                        inputs=images,
                        predictor=tta_model,
                        sw_batch_size=32,
                        roi_size=(800, 800),
                        overlap=0.25,
                        padding_mode="reflect",
                        mode="gaussian",
                        sw_device="cuda",
                        device="cuda",
                        progress=False,
                    )
                if axis == "X":
                    y_preds[idx, :, :] += (
                        (preds / sum(weights)).squeeze().sigmoid().cpu().numpy() / 3.0
                    ).astype(np.half)
                elif axis == "Y":
                    y_preds[:, idx, :] += (
                        (preds / sum(weights)).squeeze().sigmoid().cpu().numpy() / 3.0
                    ).astype(np.half)
                elif axis == "Z":
                    y_preds[:, :, idx] += (
                        (preds / sum(weights)).squeeze().sigmoid().cpu().numpy() / 3.0
                    ).astype(np.half)

            # np.save(f"{DATASET_FOLDER}/{folder}.npy", y_preds)

            for i, pred in tqdm(enumerate(y_preds)):
                cv2.imwrite(
                    f'{DATASET_FOLDER}/{folder}/labels/{ls_images[i].split("/")[-1].replace(".jp2", ".png")}',
                    (pred * 255).astype(np.uint8),
                )

            # del test_dataset, test_loader, y_preds
            # gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation Model Inference")
    parser.add_argument(
        "--dataset_folder", type=str, required=True, help="Path to the dataset folder"
    )
    parser.add_argument(
        "--logs_base_path",
        type=str,
        required=True,
        help="Path to the logs base directory",
    )
    args = parser.parse_args()
    DATASET_FOLDER = args.dataset_folder
    logs_base_path = args.logs_base_path
