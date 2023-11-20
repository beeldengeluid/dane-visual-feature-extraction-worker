import numpy as np
import matplotlib.pyplot as plt
import os
import torch

spectogram_path = '/home/longteng/code/dane-visual-feature-extraction-worker/data/input-files/test_source_id/spectograms/5.npz'
save_path = 'spectogram_test.png'

data = np.load(spectogram_path, allow_pickle=True)
audio = data['arr_0'].item()['audio']
audio = torch.tensor(audio).squeeze()

plt.imshow(audio.T)
plt.savefig(save_path)