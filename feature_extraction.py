import logging
from nn_models import load_model_from_file
import sys
import torch
import numpy as np
import os
from pathlib import Path
from data_handling import VisXPData


logger = logging.getLogger(__name__)


def extract_features(input_path: str, source_id, model_path, model_config_file):
    # consult misc/feature_examples/feat_demo.py !!
    source_path = Path(input_path) / source_id
    # Load spectograms + keyframes from file & preprocess
    dataset = VisXPData(
        source_path, model_config_file=model_config_file
    )

    # Load model from file
    model = load_model_from_file(
        checkpoint_file=model_path,
        config_file=model_config_file,
    )

    # Apply model to data
    logger.info(f"Going to extract features for {dataset.__len__()} items. ")

    result = None
    for i, batch in enumerate(dataset.batches(batch_size=256)):
        frames, spectograms = batch["video"], batch["audio"]
        timestamps, shots = batch["timestamp"], batch["shot_boundaries"]
        with torch.no_grad():  # Forward pass to get the features
            audio_feat = model.audio_model(spectograms)
            visual_feat = model.video_model(frames)
        batch_result = torch.concat(
            (timestamps.unsqueeze(1), shots, audio_feat, visual_feat),
            1)
        if not result:
            result = batch_result
        else:
            result = torch.concat((result, batch_result), 0)
    export_features(result,
                    destination=os.path.join('data/visxp_features', f'{source_id}.pt'))

    # Binarize resulting feature matrix
    # Use GPU for processing
    # Store binarized feature matrix to file


def export_features(features: torch.Tensor, destination: str):
    with open(os.path.join(destination), 'wb') as f:
        torch.save(obj=features, f=f)





if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,  # configure a stream handler only for now (single handler)
        format="%(asctime)s|%(levelname)s|%(process)d|%(module)s"
        "|%(funcName)s|%(lineno)d|%(message)s",
    )

    data_path = "data/visxp_prep"
    # data_path = "data/proper_data"

    extract_features(input_path=data_path,
                     source_id='ZQWO_DYnq5Q_000000',
                     model_config_file='models/model_config.yml',
                     model_path='models/checkpoint.tar')


def example_function():
    return 0 == 0
