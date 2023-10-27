import feature_extraction
import os
import torch
import numpy as np
import logging

def test_extract_features():
    
    #assert True
    #return
    
    feature_extraction.extract_features(
        input_path='tests/data',
        model_path='models/checkpoint.tar',
        model_config_file='models/model_config.yml',
        output_path='tests/data/')
    feature_file = 'tests/data/data.pt'
    with open(feature_file, 'rb') as f:
        features = torch.load(f)
    features = features[:, 3:]  # columns 0,1,2 hold timestamps & shot boundaries
    example_features = torch.Tensor(np.load('tests/data/demo_concat_feat.npy'))
    
    


    assert torch.equal(features, example_features)

    os.remove(feature_file)
    assert feature_extraction.example_function()
