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
    def __init__(self, datapath: Path, model_config_file: str, check_spec_dim=False):
        if type(datapath) is not Path:
            datapath = Path(datapath)
        # Sorting not really necessary, but is a (poor) way of making sure specs and frames are aligned..
        self.spec_paths = sorted(list(datapath.glob("spectograms/*.npz")))
        self.frame_paths = sorted(list(datapath.glob("keyframes/*.jpg")))
        self.set_config(model_config_file=model_config_file)
        self.list_of_shots = self.ListOfShots(datapath)
        if check_spec_dim:
            all_ok = self.check_spec_dim()
            if all_ok:
                logger.info(
                    "Sampled pectogram dimensionalities match specification"
                    f" in model config({self.dim_a})"
                )

    def check_spec_dim(self, sample_size=5):
        all_ok = True
        for index in random.sample(range(len(self.spec_paths)), sample_size):
            spec = self.__get_spec__(index=index, transform=False)
            try:
                assert spec.shape == self.dim_a
            except AssertionError:
                logger.info(
                    f"Spectogram dimensionality ({spec.shape})"
                    f" does not match specification in "
                    f"model config({self.cfg.SPECTOGRAM.DIMENSIONALITY})"
                )
                all_ok = False
        return all_ok

    def set_config(self, model_config_file: str):
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

    def __len__(self):
        return len(self.spec_paths)

    def __getitem__(self, index):
        item_dict = dict()
        item_dict["video"] = self.__get_keyframe__(index=index)
        item_dict["audio"] = self.__get_spec__(index=index)
        timestamp = int(
            self.frame_paths[index].parts[-1].split(".")[0]
        )  # TODO: set proper timestamp and make sure audio and video are actually aligned
        item_dict["timestamp"] = timestamp
        item_dict["shot_boundaries"] = self.list_of_shots.find_shot_for_timestamp(
            timestamp=timestamp
        )
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

    def batches(self, batch_size: int = 1):
        return DataLoader(self, batch_size=batch_size, shuffle=False)

    class ListOfShots:
        def __init__(self, datapath: Path):
            with open(
                datapath / "metadata" / "shot_boundaries_timestamps_ms.txt", "r"
            ) as f:
                self.list_of_tuples = np.array(eval(f.read()))

        def find_shot_for_timestamp(self, timestamp):
            a = self.list_of_tuples
            hits = a[np.logical_and(a[:, 0] <= timestamp, a[:, 1] >= timestamp)]
            if len(hits != 0):
                # matched one (or multiple) shots: return boundaries (of the first hit)
                return hits[0]
            else:
                # No corresponding shot (shots may be un-covering). Signal with (-1,-1)
                return np.array((-1, -1))
