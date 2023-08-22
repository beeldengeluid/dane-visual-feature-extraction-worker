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
    # Debuging
    #############################################################################
    video_bad_size_inx = [item['video'].size() != torch.Size([3, 30, 224, 224]) for item in batch]
    video_bad_size_inx = [False] * len(batch) # debugging
    if sum(video_bad_size_inx) > 0:
        for i, inx in enumerate(video_bad_size_inx):
            if inx == True:
                video_path = batch[i]['video_name']
                logger.error(f'wrong shaped video {video_path}, skipping...')

    audio_bad_size_inx = [item['audio'].size() != torch.Size([1, 257, 99]) for item in batch]
    if sum(audio_bad_size_inx) > 0:
        for i, inx in enumerate(audio_bad_size_inx):
            if inx == True:
                video_path = batch[i]['video_name']
                logger.error(f'wrong shaped audio {video_path}, skipping ...')

    good_size_inx = [not (video_bad_size_inx[i] or audio_bad_size_inx[i]) for i in range(len(video_bad_size_inx))]
    batch = [batch[i] for i in range(len(good_size_inx)) if good_size_inx[i]]
    #############################################################################
    batch = list(filter(lambda x:x is not None, batch))

    if len(batch) == 0:
        return None
    else:
        return default_collate(batch)

class VggSoundDataModule(pytorch_lightning.LightningDataModule):

    def __init__(self,
                    data_path = '/local/tlong/data/kinetics_sound_split', # Dataset configuration
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
        Create the Kinetics validation partition from the list of video labels
        in {self._DATA_PATH}/{split}}, sample_mode can be 'random' or 'uniform'
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


if __name__ == '__main__':
    data_module = VggSoundDataModule(num_workers = 70, clip_duration = 1, clip_frames = 30, batch_size = 1)

    train_loader = data_module.get_loader('train', 'random')
    train_loader_uniform = data_module.get_loader('train', 'uniform')
    valid_loader = data_module.get_loader('val', 'uniform')

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
