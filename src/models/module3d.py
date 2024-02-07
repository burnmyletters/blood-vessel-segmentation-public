from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric
from torchmetrics.classification import Dice
import numpy as np
from src.utils.metric import fast_compute_surface_dice_score_from_tensor
import os


class LitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss: torch.nn.modules.loss._Loss,
        compile: bool,
        output_path: str,
        img_size: Tuple[int, int] = (192, 192, 192),
        surface_dice_calculate: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["net", "loss"], logger=False)

        self.net = net
        self.criterion = loss

        self.train_metric = Dice()
        self.val_metric = Dice()
        self.preds, self.targets = [], []

        # self.val_surface_dice = SurfaceDiceMetric()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.current_step = 0

        self.metric_save = 0
        self.sanity = True
        os.makedirs(self.hparams.output_path, exist_ok=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        # self.val_surface_dice.reset()
        self.val_metric.reset()

    def model_step(
        self,
        batch: Dict,
        loader: str,
        batch_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch["volume"], batch["target"]
        logits = self.forward(x)

        # if loader == "train":
        #     logits = self.forward(x)
        #
        # else:
        #     logits = monai.inferers.sliding_window_inference(
        #         inputs=x,
        #         predictor=self.net,
        #         sw_batch_size=8,
        #         roi_size=self.hparams.img_size,
        #         overlap=0.25,
        #         padding_mode="reflect",
        #         mode="gaussian",
        #         sw_device="cuda",
        #         device="cuda",
        #         progress=False,
        #     )

        loss = self.criterion(logits, y)

        preds = logits.sigmoid()
        if loader != "test":
            if batch_idx % 200 == 0:
                self.log_image(img=x, y_pred=preds, seg=y, stage=loader)

        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(
            batch, batch_idx=batch_idx, loader="train"
        )

        # update and log metrics
        self.train_loss(loss)
        self.train_metric(preds, targets.long())
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/dice", self.train_metric, on_step=False, on_epoch=True, prog_bar=True
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch, batch_idx=batch_idx, loader="val")
        # update and log metrics
        self.val_loss(loss)
        self.val_metric(preds, targets.long())

        if self.hparams.surface_dice_calculate:
            if isinstance(self.targets, list):
                self.targets.append(targets.detach().squeeze().cpu().numpy())
                self.preds.append(
                    preds.detach().squeeze().cpu().numpy().astype(np.half)
                )
            else:
                self.targets[batch_idx, :, :, :] = (
                    targets.detach().squeeze().cpu().numpy()
                )
                self.preds[batch_idx, :, :, :] = (
                    preds.detach().squeeze().cpu().numpy().astype(np.half)
                )

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/dice", self.val_metric, on_step=False, on_epoch=True, prog_bar=True
        )
        # self.log(
        #     "val/surface_dice", self.val_surface_dice, on_step=False, on_epoch=True, prog_bar=True
        # )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."

        # try:
        #     self.net.encoder.set_swish(memory_efficient=False)
        # except:
        #     pass
        #
        # self.to_torchscript(
        #     file_path=f"{self.hparams.output_path}/model_trace.pt",
        #     method="trace",
        #     example_inputs=torch.randn(1, self.hparams.in_channels, 512, 512),
        # )
        # try:
        #     self.net.encoder.set_swish(memory_efficient=True)
        # except:
        #     pass

        if self.hparams.surface_dice_calculate:
            if isinstance(self.targets, list):
                self.targets = np.array(self.targets)
                self.preds = np.array(self.preds)

            for th in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
                metric = fast_compute_surface_dice_score_from_tensor(
                    (self.preds > th).astype(np.uint8), self.targets
                )

                self.log(
                    f"val/sd@{th}",
                    metric,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )

            if self.sanity:
                self.targets = []
                self.preds = []
                self.sanity = False

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        pass

    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # "monitor": "val/loss",
                    "interval": "step",  # "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def log_image(self, img, y_pred, seg, stage="train"):
        log_image = self.get_visuals(img, y_pred, seg)
        if log_image is not None:
            self.logger.experiment.add_image(
                tag=f"{stage}_viz",
                img_tensor=log_image,
                global_step=self.current_step,
                dataformats="HWC",
            )
        self.current_step += 100

    @staticmethod
    def get_visuals(inputs, outputs, target):
        res_img = None

        target = target.detach().cpu().numpy()
        inputs = inputs.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()

        for idx in range(min(inputs.shape[0], 4)):
            viz = []
            for ax in range(3):
                img = np.stack(
                    ((np.mean(inputs[idx, ...].squeeze(), axis=ax)),) * 3, axis=-1
                )
                img = (255 * img).astype("uint8")
                gt = np.stack(
                    ((np.mean(target[idx, ...].squeeze(), axis=ax)),) * 3, axis=-1
                )
                gt /= gt.max()
                gt = (255 * gt).astype("uint8")
                pred = np.stack(
                    ((np.mean(outputs[idx, ...].squeeze(), axis=ax)),) * 3, axis=-1
                )
                pred /= pred.max()
                pred = (255 * pred).astype("uint8")

                pred_th = np.stack(
                    ((np.mean(1.0 * (outputs[idx, ...].squeeze() > 0.5), axis=ax)),)
                    * 3,
                    axis=-1,
                )
                pred_th /= pred_th.max()
                pred_th = (255 * pred_th).astype("uint8")

                # viz.append(np.hstack([img, gt, pred, dif]))
                viz.append(np.hstack([img, gt, pred, pred_th]))

            viz = np.hstack(viz)
            res_img = np.vstack((res_img, viz)) if res_img is not None else viz

        return res_img


if __name__ == "__main__":
    model = LitModule(None, None, None, None)
