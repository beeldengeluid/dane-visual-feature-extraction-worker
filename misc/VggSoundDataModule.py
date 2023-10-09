import pytorch_lightning
import pytorchvideo.data
import pytorchvideo.transforms
import torch
import torch.utils.data
import os, logging
from torch.utils.data import default_collate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def my_collate(batch):
    batch = list(filter(lambda x:x is not None, batch))

    if len(batch) == 0:
        return None
    else:
        return default_collate(batch)

class VggSoundDataModule(pytorch_lightning.LightningDataModule):

    def __init__(self,
                    data_path = './data', # Dataset configuration
                    clip_duration = 16/30, # Duration of sampled clip for each video
                    batch_size = 8,
                    num_workers = 0, # Number of parallel processes fetching data
                    clip_frames = 16,
                    transform = None
                ):

        super().__init__()

        self._DATA_PATH = data_path 
        self._CLIP_DURATION = clip_duration
        self._BATCH_SIZE = batch_size
        self._NUM_WORKERS = num_workers  
        self._CLIP_FRAMES = clip_frames

        if transform is not None:
            self._TRANSFORM = transform
        else:
            self._TRANSFORM = None

    def get_loader(self, split, sample_mode):
        """ 
        Args:
            split: 'train' or 'val'
            sample_mode: 'random' or 'uniform'
        """

        dataset = pytorchvideo.data.labeled_video_dataset(
            data_path=os.path.join(self._DATA_PATH, split),
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                sample_mode, self._CLIP_DURATION),
            decode_audio=True,
            transform=self._TRANSFORM)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            # persistent_workers=True,
            shuffle=False,
            drop_last=True,
            collate_fn=my_collate
        )

        return dataloader
    
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--clip_duration', type=float, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--clip_frames', type=int, default=30)
parser.add_argument('--sample_mode_train', type=str, default='random')
parser.add_argument('--sample_mode_test', type=str, default='uniform')
parser.add_argument('--transform', type=str, default=None)
args = parser.parse_args()


if __name__ == '__main__':
    data_module = VggSoundDataModule(num_workers = args.num_workers, 
                                     data_path = args.data_path, 
                                     clip_duration = args.clip_duration, 
                                     batch_size = args.batch_size, 
                                     clip_frames = args.clip_frames)

    train_loader = data_module.get_loader('train', args.sample_mode_train)
    train_loader_uniform = data_module.get_loader('train', args.sample_mode_test)
    valid_loader = data_module.get_loader('val', args.sample_mode_test)

    for batch in train_loader:
        break

    video_index_list = []

    # dict_keys(['video', 'video_name', 'video_index', 'clip_index', 'aug_index', 'label', 'audio'])
    for i, batch in enumerate(train_loader_uniform):

        if batch == None:
            continue

        video, audio = batch['video'], batch['audio']
        video_index, clip_index = batch['video_index'], batch['clip_index']
        video_name, label = batch['video_name'], batch['label']
        
        print(i, video_index, clip_index)
        video_index_list.append(video_index)
