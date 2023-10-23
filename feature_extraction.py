import logging
from nn_models import load_model_from_file
import sys
import torch
import os

from dane.config import cfg
from data_handling import VisXPData


logger = logging.getLogger(__name__)


def extract_features(input_path: str):
    # consult misc/feature_examples/feat_demo.py !!

    # Load spectograms + keyframes from file & preprocess
    dataset = VisXPData(
        input_path, model_config_file=cfg.VISXP_EXTRACT.MODEL_CONFIG_PATH
    )

    # Load model from file
    model = load_model_from_file(
        checkpoint_file=cfg.VISXP_EXTRACT.MODEL_PATH,
        config_file=cfg.VISXP_EXTRACT.MODEL_CONFIG_PATH,
    )

    # Apply model to data
    logger.info(f"Going to extract features for {dataset.__len__()} items. ")

    audio_feat, visual_feat = torch.tensor([]), torch.tensor([])

    for i, batch in enumerate(dataset.batches(batch_size=2)):
        frames, spectograms = batch["video"], batch["audio"]
        sourceIDs, timestamps = batch["videoname"],batch["timestamp"]
        with torch.no_grad():  # Forward pass to get the features
            audio_feat = model.audio_model(spectograms)
            visual_feat = model.video_model(frames)
        logger.info(
            "Extracted features. "
            f"Audio shape: {audio_feat.shape}, "
            f"visual shape: {visual_feat.shape}"
        )
        # TODO: prepend matrix with shot boundaries and timestamp, and then save to file per sourceID

    # Binarize resulting feature matrix
    # Use GPU for processing
    # Store binarized feature matrix to file


def export_features(audio_feat: torch.Tensor, visual_feat: torch.Tensor, destination: str):
    with open(os.path.join(destination, 'audio_features'), 'wb') as f:
        torch.save(obj=audio_feat, f=f)





if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,  # configure a stream handler only for now (single handler)
        format="%(asctime)s|%(levelname)s|%(process)d|%(module)s"
        "|%(funcName)s|%(lineno)d|%(message)s",
    )

    data_path = "data/visxp_prep"
    # data_path = "data/proper_data"

    extract_features(data_path)


def example_function():
    return 0 == 0
