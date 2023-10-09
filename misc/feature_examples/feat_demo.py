import argparse
from pathlib import Path
from L3_data_module import L3_data_module
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from models import AVNet
import os

parser = argparse.ArgumentParser(description='L3')

parser.add_argument('--train_path', type=str, default='/local/tlong/data/vggsound_l3/train/', metavar='N',
                    help='path to train data (default: /local/tlong/data/kineticssound_l3/train/)')
parser.add_argument('--test_path', type=str, default='/local/tlong/data/vggsound_l3/val/', metavar='N',
                    help='path to test data (default: /local/tlong/data/kineticssound_l3/test/)')
parser.add_argument('--num_classes', type=int, default=339)
parser.add_argument('--double_convolution', type=bool, default=True, metavar='N',
                    help='double convolution (default: True)')
parser.add_argument('--ckpt_path', type=str, default='./L3_kinetics_double_conv_rebuttal/checkpoints/checkpoint_7.0.pth.tar')

parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--num_workers', type=int, default=80, metavar='N',
                    help='number of workers (default: 80)')

args = parser.parse_args()
args.train_path = Path(args.train_path)
args.test_path = Path(args.test_path)

def load_checkpoint(args, model):

    checkpoint = torch.load(args.ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])

class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        
        self.features = original_model.features
        self.classifier = nn.Sequential(*list(original_model.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x

if __name__ == '__main__':

    # test_kmeans()
    model = AVNet(num_classes = 2, double_convolution=args.double_convolution)
    load_checkpoint(args, model)

    test_dataset = L3_data_module(args.test_path, mode='normal')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, collate_fn=None)

    model.eval()
    with torch.no_grad():

        for i, batch in enumerate(test_loader):
            if batch is None:
                continue
            frame, audio, label, avlabel = batch['video'], batch['audio'], batch['label'], batch['avlabel']
            cls_name, videoname, index = batch['cls_name'], batch['videoname'], batch['index']

            # Forward pass to get the features
            audio_feat = model.audio_model(audio)
            visual_Feat = model.video_model(frame)