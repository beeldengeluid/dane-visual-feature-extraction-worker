import numpy as np
import torch
import torch.nn.functional as F
from VggSoundDataModule import VggSoundDataModule
from PIL import Image

def save_image(frame_tensor : torch.FloatTensor, save_path):
    # frame_tensor: 
    # shape: (3, W, H), 
    # range: [0, 255], 
    # type: float

    numpy_uint8_frame = frame_tensor.permute(1, 2, 0).numpy().astype('uint8')
    frame = Image.fromarray(numpy_uint8_frame, 'RGB')
    frame.save(save_path)


from tqdm import tqdm
from pathlib import Path

class ARGS():
    def __init__(self):
        # data parameters
        # self.dataset                    = 'kinetics'
        # self.data_dir                   = '/local/tlong/data/vggsound_by_cls'
        self.data_dir                   = '/local/tlong/data/kinetics_sound_split'
        self.mode                       = 'test' # mode here means train/val/test split

        self.num_sec_aud                = 1 # default
        self.aud_sample_rate            = 24000 # default
        self.aud_spec_type              = 2
        self.use_volume_jittering       = False # default
        self.use_audio_temp_jittering   = False # default
        self.z_normalize                = True  # default for audio

        self.batch_size                 = 1
        self.workers                    = 70

        # l3 parameters
        # self.l3_input_base     = Path('/local/tlong/data/vggsound_l3/')
        self.l3_input_base     = Path('/local/tlong/data/kineticssound_l3/')



def get_clip_store_path(feat_base, video_name, clip_index, i):

    path_parts = Path(video_name[i]).parts
    class_name, video_fn = path_parts[-2], path_parts[-1], 
    video_base = feat_base / class_name / video_fn

    video_base.mkdir(parents=True, exist_ok=True)

    # output file name is defined as {video_name}/{clip_index}.npz
    clip_feat_fn_wo_ext = video_base / str(clip_index[i].item())
    clip_feat_fn_wi_ext = clip_feat_fn_wo_ext.with_suffix('.npz')
    return clip_feat_fn_wi_ext


if __name__ == '__main__':

    # Step0 : Degine args and data module
    ################################################################################################
    args = ARGS()
    data_module = VggSoundDataModule(args.data_dir,
                                 batch_size = args.batch_size,
                                 num_workers = args.workers, clip_duration = 1, clip_frames = 30)

    train_loader = data_module.get_loader('train', 'random')
    train_loader_uniform = data_module.get_loader('train', 'uniform')
    valid_loader = data_module.get_loader('val', 'uniform')
    ################################################################################################

    l3_input_base = args.l3_input_base / args.mode
    l3_input_base.mkdir(parents=True, exist_ok=True)

    if args.mode == 'train':
        loader = train_loader_uniform
    else:
        loader = valid_loader

    with torch.no_grad():

        for batch_idx, batch in tqdm(enumerate(loader)): 

            if batch is None:
                continue
            if 'audio' not in batch.keys():
                continue

            frames, audios, labels = batch['video'], batch['audio'], batch['label']
            video_name, video_index, clip_index = batch['video_name'], batch['video_index'], batch['clip_index']

            # frames, audios, labels = frames.cuda(), audios.cuda(), labels.cuda()
            # Step3.1: check if entire batch are already extracted, to avoid forward() 
            ################################################################################################
            exist_samples_in_batch = set()
            for i in range(len(labels)):
                
                clip_store_path = get_clip_store_path(l3_input_base, video_name, clip_index, i)

                if clip_store_path.exists():
                    # print(f'video {video_name[i]}, clip {clip_index[i]} feature exists')
                    exist_samples_in_batch.add(i)
                    continue

            if len(exist_samples_in_batch) == len(labels):
                # print(f'skip batch for video {video_name[i]} as feature exists')
                continue
            ################################################################################################

            # Step3.2 get the frame and audios.
            ################################################################################################

            for i in range(len(labels)):

                audio_path = get_clip_store_path(l3_input_base, video_name, clip_index, i)
                frame_path = audio_path.with_suffix('.jpg')
                frame = frames[i, :, 0, :, :]

                audio = audios[i,:,:,:].numpy()

                out_feat_dict = dict()
                out_feat_dict['audio'] = audio

                np.savez(audio_path, out_feat_dict) # type: ignore
                save_image(frame, frame_path)

                # import pdb
                # pdb.set_trace()
