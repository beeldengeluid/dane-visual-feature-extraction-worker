from torch.utils.data import Dataset, DataLoader, default_collate

import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

import numpy as np
from glob import glob
from pathlib import Path
from time import time


class L3_data_module(Dataset):

    def __init__(self, l3_path: Path, mode: str):

        st = time()
        self.mode = mode

        self.l3_clips = list(l3_path.glob('*/*/*.npz'))

        self.visual_transform = T.Compose(
            [   
                T.Normalize((0.3961, 0.3502, 0.3224),
                            (0.2298, 0.2237, 0.2197)),
                T.Resize((256, 256), antialias=True),
            ])

        self.audio_transform = T.Compose(
            [
                T.Normalize((0.0467), (0.9170)),
            ])

        self.label_set = [item.parts[-1] for item in l3_path.glob('*')]
        self.label_set.sort()
        print(time() - st, len(self.l3_clips))

    def __len__(self):

        return len(self.l3_clips)

    def get_pair(self, audio_path, frame_path, avlabel, index):

        frame = torchvision.io.read_image(frame_path) / 255.0
        frame = self.visual_transform(frame)

        data = np.load(audio_path, allow_pickle=True)
        audio = data['arr_0'].item()['audio']
        audio = torch.tensor(audio)
        audio = self.audio_transform(audio)

        cls_name = audio_path.parts[2]
        label = self.label_set.index(cls_name)

        batch_dict = dict()
        if avlabel:
            batch_dict['video'], batch_dict['audio'], batch_dict['label'] = frame, audio, label
            batch_dict['avlabel'] = 1
            batch_dict['cls_name'], batch_dict['videoname'], batch_dict['index'] = cls_name, audio_path.parts[3], index
        else:
            batch_dict['video'], batch_dict['audio'], batch_dict['label'] = frame, audio, -1
            batch_dict['avlabel'] = 0
            batch_dict['cls_name'], batch_dict['videoname'], batch_dict['index'] = 'AV_negative', 'AV_negative_frame', -1

        return batch_dict

    def get_positive_pairs(self, index):
            
        audio_path = self.l3_clips[index]
        frame_path = str(audio_path.with_suffix('.jpg'))
        avlabel = True

        batch_dict = self.get_pair(audio_path, frame_path, avlabel, index)

        return batch_dict
    
    def get_negative_pairs(self, index):

        audio_path = self.l3_clips[index]
        N = len(self.l3_clips)
        neg_index = torch.randint(0, N, (1,))[0]
        while neg_index == index:
            neg_index = torch.randint(0, N, (1,))[0]
        frame_path = str(self.l3_clips[neg_index].with_suffix('.jpg'))
        avlabel = False

        batch_dict = self.get_pair(audio_path, frame_path, avlabel, index)

        return batch_dict

    def __getitem__(self, index):

        if self.mode == 'AV':
            if torch.rand(1) > 0.5:
                batch_dict = self.get_negative_pairs(index)
            else:
                batch_dict = self.get_positive_pairs(index)
        elif self.mode == 'normal':
            batch_dict = self.get_positive_pairs(index)
        else:
            raise ValueError('mode should be AV or normal')

        return batch_dict

class L3_data_resampled_module(L3_data_module):

    def __init__(self, l3_path: Path, mode: str, resample_indexes, pseudo_labels):

        super().__init__(l3_path, mode)
        self.l3_clips_re = [self.l3_clips[i] for i in resample_indexes]
        self.videonames_re = [item.parts[7] for item in self.l3_clips_re]
        self.pseudo_labels = pseudo_labels
        self.pseudo_labels_re = pseudo_labels[resample_indexes]

    def __getitem__(self, index):
                
        audio_path = self.l3_clips_re[index]
        frame_path = str(audio_path.with_suffix('.jpg'))
        plabel = self.pseudo_labels_re[index]

        frame = torchvision.io.read_image(frame_path) / 255.0
        frame = self.visual_transform(frame)

        data = np.load(audio_path, allow_pickle=True)
        audio = data['arr_0'].item()['audio']
        audio = torch.tensor(audio)
        audio = self.audio_transform(audio)

        cls_name = audio_path.parts[6]
        label = self.label_set.index(cls_name)

        batch_dict = dict()

        batch_dict['video'], batch_dict['audio'], batch_dict['label'] = frame, audio, label
        batch_dict['pseudo_label'] = plabel
        batch_dict['cls_name'], batch_dict['videoname'] = cls_name, audio_path.parts[7]

        return batch_dict
    
if __name__ == '__main__':

    train_path = Path('/local/tlong/data/vggsound_l3/train/')
    train_dataset = L3_data_module(train_path, mode = 'normal')

    train_loader = DataLoader(train_dataset, batch_size = 16, 
                                    shuffle = True,
                                    num_workers = 0,
                                    drop_last = True,
                                    collate_fn = None)

    for batch in train_loader:

        if batch is None:
            continue

        frame, audio, label, avlabel = batch['video'], batch['audio'], batch['label'], batch['avlabel']
        cls_name, videoname, index = batch['cls_name'], batch['videoname'], batch['index']

        print(frame.shape, audio.shape, label, avlabel)
        print(cls_name, videoname, index)

        import pdb
        pdb.set_trace()

        break

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)