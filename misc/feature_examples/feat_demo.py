import argparse
from pathlib import Path

import numpy as np
from L3_data_module import L3_data_module
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from models import AVNet
import os

parser = argparse.ArgumentParser(description='L3')

parser.add_argument('--train_path', type=str, default='../data/train/', metavar='N',
                    help='path to train data (default: /local/tlong/data/kineticssound_l3/train/)')
parser.add_argument('--test_path', type=str, default='../frame_and_spectogram_examples', metavar='N',
                    help='path to test data (default: /local/tlong/data/kineticssound_l3/test/)')
parser.add_argument('--num_classes', type=int, default=339)
parser.add_argument('--double_convolution', type=bool, default=True, metavar='N',
                    help='double convolution (default: True)')
parser.add_argument('--ckpt_path', type=str, default='../../models/checkpoint.tar')

parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                    help='number of workers (default: 80)')

args = parser.parse_args()
args.train_path = Path(args.train_path)
args.test_path = Path(args.test_path)

def load_checkpoint(args, model):

    if torch.cuda.is_available():
        checkpoint = torch.load(args.ckpt_path)
    else:
        checkpoint = torch.load(args.ckpt_path,  map_location=torch.device('cpu'))

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
        concat_feat_list = []

        for i, batch in enumerate(test_loader):
            if batch is None:
                continue
            frame, audio, label, avlabel = batch['video'], batch['audio'], batch['label'], batch['avlabel']
            cls_name, videoname, index = batch['cls_name'], batch['videoname'], batch['index']
            original_index = batch['original_index'].unsqueeze(1)
            #import pdb
            #pdb.set_trace()
            # Forward pass to get the features
            audio_feat = model.audio_model(audio)
            visual_Feat = model.video_model(frame)

            # Concatenate the features
            concat_feat = torch.cat((original_index, audio_feat, visual_Feat), 1)


            # Save the features
            
            concat_feat_list.append(concat_feat)

            print('Processing batch {} / {}'.format(i, len(test_loader)))

        # Save the features
        concat_feat, _ = torch.cat(concat_feat_list).sort(0)
        with open('demo_concat_feat.pt', 'wb') as f:
            torch.save(concat_feat, f)