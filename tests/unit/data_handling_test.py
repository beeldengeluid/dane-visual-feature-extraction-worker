from data_handling import VisXPData
import torch
import torchvision.transforms as T
import torchvision
import numpy as np


def test_batches():
    dataset = VisXPData('tests/data/', model_config_file='models/model_config.yml')
    for i, item in enumerate(dataset.batches(1)):
        index = int(item['timestamp'][0])
        audio = item['audio'][0]
        audio_example = obtain_example_audio(index)
        assert torch.equal(audio_example, audio)

        video = item['video'][0]
        video_example = obtain_example_video(index)
        assert torch.equal(video_example, video)


def obtain_example_video(i):
    frame_path = f'tests/data/keyframes/{i}.jpg'
    frame = torchvision.io.read_image(frame_path) / 255.0
    visual_transform = T.Compose(
            [   
                T.Normalize((0.3961, 0.3502, 0.3224),
                            (0.2298, 0.2237, 0.2197)),
                T.Resize((256, 256), antialias=True),
            ])
    frame = visual_transform(frame)
    return frame


def obtain_example_audio(i):
    data = np.load(f'tests/data/spectograms/{i}.npz', allow_pickle=True)
    audio = data['arr_0'].item()['audio']
    audio = torch.tensor(audio)
    audio_transform = T.Compose(
            [
                T.Normalize((0.0467), (0.9170)),
            ])
    audio = audio_transform(audio)
    return audio
