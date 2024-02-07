from albumentations import ChannelDropout as CD
from typing import Mapping, Any
import random


class ChannelDropout(CD):
    def get_params_dependent_on_targets(self, params: Mapping[str, Any]):
        img = params["image"]

        num_channels = img.shape[-1]

        if len(img.shape) == 2 or num_channels == 1:
            raise NotImplementedError(
                "Images has one channel. ChannelDropout is not defined."
            )

        if self.max_channels >= num_channels:
            raise ValueError("Can not drop all channels in ChannelDropout.")

        num_drop_channels = random.randint(self.min_channels, self.max_channels)

        flag = True
        while flag:
            channels_to_drop = random.sample(range(num_channels), k=num_drop_channels)
            if channels_to_drop != num_channels // 2:
                flag = False

        return {"channels_to_drop": channels_to_drop}


if __name__ == "__main__":
    from albumentations import Compose
    import numpy as np

    transform = Compose([ChannelDropout(p=1.0)])

    img = np.random.rand(4, 4, 3)
    data = transform(image=img)["image"]

    print(data)
