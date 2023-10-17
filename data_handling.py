from torch.utils.data import Dataset, DataLoader  # type: ignore
import torch  # type: ignore
import torchvision  # type: ignore
import torchvision.transforms as T  # type: ignore
import numpy as np
from pathlib import Path


class VisXPData(Dataset):
    def __init__(self, datapath: Path):
        if type(datapath) is not Path:
            datapath = Path(datapath)
        self.spec_paths = list(datapath.glob("*/spectograms/*.npz"))
        self.frame_paths = list(datapath.glob("*/keyframes/*.jpg"))

        self.visual_transform = T.Compose(
            [
                T.Normalize((0.3961, 0.3502, 0.3224), (0.2298, 0.2237, 0.2197)),
                T.Resize((256, 256), antialias=True),
            ]
        )

        self.audio_transform = T.Compose(
            [
                T.Normalize((0.0467), (0.9170)),
            ]
        )

    def __len__(self):
        return len(self.spec_paths)

    def __getitem__(self, index):

        frame = torchvision.io.read_image(str(self.frame_paths[index])) / 255.0
        frame = self.visual_transform(frame)

        data = np.load(self.spec_paths[index], allow_pickle=True)
        audio = data['arr_0'].item()['audio']
        audio = torch.tensor(audio)
        audio = self.audio_transform(audio)

        item_dict = dict()
        item_dict["video"] = frame
        item_dict["audio"] = audio
        item_dict["videoname"] = self.frame_paths[index].parts[0]
        item_dict["timestamp"] = self.frame_paths[index].parts[-1].split(".")[0]

        return item_dict

    def batches(self):
        return DataLoader(self)
