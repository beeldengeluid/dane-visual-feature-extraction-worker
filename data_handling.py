from torch.utils.data import Dataset, DataLoader  # type: ignore
import torch  # type: ignore
import torchvision  # type: ignore
import torchvision.transforms as T  # type: ignore
import numpy as np
from pathlib import Path
from yacs.config import CfgNode as CN
import random
import logging

logger = logging.getLogger(__name__)


class VisXPData(Dataset):
    def __init__(
        self, datapath: str | Path, model_config_file: str, check_spec_dim=False
    ):
        if type(datapath) is not Path:
            datapath = Path(datapath)
        self.spec_paths = list(datapath.glob("*/spectograms/*.npz"))
        self.frame_paths = list(datapath.glob("*/keyframes/*.jpg"))

        with open(model_config_file, "r") as f:
            cfg = CN.load_cfg(f).INPUT
            norm_a = eval(cfg.SPECTOGRAM.NORMALIZATION)
            self.dim_a = eval(cfg.SPECTOGRAM.DIMENSIONALITY)
            self.audio_transform = T.Compose(
                [
                    T.Normalize(norm_a[0], norm_a[1]),
                ]
            )
            norm_v = eval(cfg.KEYFRAME.NORMALIZATION)
            dim_v = eval(cfg.KEYFRAME.DIMENSIONALITY)
            self.visual_transform = T.Compose(
                [
                    T.Normalize(norm_v[0], norm_v[1]),
                    T.Resize(dim_v, antialias=True),
                ]
            )

        if check_spec_dim:
            all_ok = True
            for index in random.sample(range(len(self.spec_paths)), 5):
                ok = self.check_spec_dim(index)
                all_ok = min(ok, all_ok)
            if all_ok:
                logger.info(
                    "Spectogram dimensionalities match specification in "
                    f"model config({self.dim_a})"
                )

    def check_spec_dim(self, index):
        spec = self.__get_spec__(index=index, transform=False)
        try:
            assert spec.shape == self.dim_a
        except AssertionError:
            logger.info(
                f"Spectogram dimensionality ({spec.shape})"
                f" does not match specification in "
                f"model config({self.cfg.SPECTOGRAM.DIMENSIONALITY})"
            )
            return False
        return True

    def __len__(self):
        return len(self.spec_paths)

    def __getitem__(self, index):
        frame = self.__get_keyframe__(index=index)
        audio = self.__get_spec__(index=index)

        item_dict = dict()
        item_dict["video"] = frame
        item_dict["audio"] = audio
        item_dict["videoname"] = self.frame_paths[index].parts[0]
        item_dict["timestamp"] = self.frame_paths[index].parts[-1].split(".")[0]

        return item_dict

    def __get_spec__(self, index, transform=True):
        data = np.load(self.spec_paths[index], allow_pickle=True)
        audio = data["arr_0"].item()["audio"]
        audio = torch.tensor(audio)
        if transform:
            audio = self.audio_transform(audio)
        return audio

    def __get_keyframe__(self, index):
        frame = torchvision.io.read_image(str(self.frame_paths[index])) / 255.0
        frame = self.visual_transform(frame)
        return frame

    def batches(self):
        return DataLoader(self)
