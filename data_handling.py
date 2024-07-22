from torch.utils.data import Dataset, DataLoader  # type: ignore
import torch  # type: ignore
import torchvision  # type: ignore
import torchvision.transforms as T  # type: ignore
import numpy as np
from pathlib import Path
from yacs.config import CfgNode as CN
import logging
from collections import defaultdict
from typing import DefaultDict
from nn_models import ModelNotFoundError

logger = logging.getLogger(__name__)


KEYFRAME_INPUT_DIR = "keyframes"
SPECTROGRAM_INPUT_DIR = "spectrograms"


class VisXPData(Dataset):
    def __init__(
        self,
        datapath: Path,
        model_config_file: str,
        device: torch.device,
    ):
        if type(datapath) is not Path:
            datapath = Path(datapath)
        self.set_config(model_config_file=model_config_file)
        self.paths: DefaultDict[int, dict] = defaultdict(dict)
        for p in datapath.glob(f"{KEYFRAME_INPUT_DIR}/*.jpg"):
            self.paths[int(p.stem)].update({"frame": p})
        self.timestamps = sorted(list(self.paths.keys()))
        self.device = device
        self.list_of_shots = self.ListOfShots(datapath)
        if self.audio_too:
            self.init_audio_too(datapath=datapath, model_config_file=model_config_file)

    def init_audio_too(self, datapath: Path, model_config_file: str):
        with open(model_config_file, "r") as f:
            cfg = CN.load_cfg(f).INPUT
        norm_a = eval(cfg.SPECTROGRAM.NORMALIZATION)
        self.dim_a = eval(cfg.SPECTROGRAM.DIMENSIONALITY)
        self.audio_transform = T.Compose(
            [
                T.Normalize(norm_a[0], norm_a[1]),
            ]
        )
        self.framerate = cfg.SPECTROGRAM.SAMPLERATE_HZ
        for p in datapath.glob(f"{SPECTROGRAM_INPUT_DIR}/*_{self.framerate}.npz"):
            self.paths[int(p.stem.split("_")[0])].update({"spec": p})

    def set_config(self, model_config_file: str):
        with open(model_config_file, "r") as f:
            cfg = CN.load_cfg(f)
        if cfg.MODEL.TYPE == 'AVNet':
            self.audio_too = True
        elif cfg.MODEL.TYPE == 'VisualNet':
            self.audio_too = False
        else:
            raise ModelNotFoundError(f"Model type '{cfg.MODEL.TYPE}' not recognized")
        norm_v = eval(cfg.INPUT.KEYFRAME.NORMALIZATION)
        self.dim_v = eval(cfg.INPUT.KEYFRAME.DIMENSIONALITY)
        self.visual_transform = T.Compose(
            [
                T.Normalize(norm_v[0], norm_v[1]),
                T.Resize(self.dim_v, antialias=True),
            ]
        )

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, index):
        item_dict = dict()
        timestamp = self.timestamps[index]
        item_dict["video"] = self.__get_keyframe__(timestamp)
        if self.audio_too:
            item_dict["audio"] = self.__get_spec__(timestamp)
        item_dict["timestamp"] = timestamp
        item_dict["shot_boundaries"] = self.list_of_shots.find_shot_for_timestamp(
            timestamp=timestamp
        )
        return item_dict

    def __get_spec__(self, timestamp: int, transform=True):
        try:
            data = np.load(str(self.paths[timestamp]["spec"]), allow_pickle=True)
            audio = data["arr_0"].item()["audio"]
            audio = torch.tensor(audio, device=self.device)
            if transform:
                audio = self.audio_transform(audio)
        except KeyError:
            logger.info(
                f"No spectrogram exists for timestamp {timestamp}"
                f" at samplerate {self.framerate}."
            )
            audio = torch.zeros(size=self.dim_a)
        return audio

    def __get_keyframe__(self, timestamp: int):
        try:
            image_file_path = str(self.paths[timestamp]["frame"])
            frame = torchvision.io.read_image(image_file_path).to(self.device)
            frame = self.visual_transform(frame / 255.0)
        except KeyError:
            logger.info(f"No keyframe exists for timestamp {timestamp}.")
            frame = torch.zeros(size=(3, self.dim_v[0], self.dim_v[1]))
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
