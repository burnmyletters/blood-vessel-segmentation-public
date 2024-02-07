import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint as PLModelCheckpoint


class ModelCheckpointWorkaround(PLModelCheckpoint):
    """Like pytorch_lightning.callbacks.ModelCheckpoint but allowing saving last top k checkpoints.
    See https://github.com/PyTorchLightning/pytorch-lightning/discussions/10669
    """

    def _validate_monitor_key(self, trainer: pl.Trainer) -> None:
        if self.monitor in (
            "step",
            "epoch",
        ):  # Lightning check in 1.5.x doesn't contemplate them.
            return
        super()._validate_monitor_key(trainer)
