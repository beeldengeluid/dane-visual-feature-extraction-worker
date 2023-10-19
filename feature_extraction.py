from nn_models import load_model_from_file
from data_handling import VisXPData
import logging
import sys
import torch

logger = logging.getLogger(__name__)


def extract_features(input_path: str):
    # consult misc/feature_examples/feat_demo.py !!

    # Load spectograms + keyframes from file & preprocess
    dataset = VisXPData(input_path, model_config_file="models/model_config.yml")

    # Load model from file
    model = load_model_from_file(
        checkpoint_file="models/checkpoint.tar",
        config_file="models/model_config.yml",
    )

    # Apply model to data
    logger.info(f"Going to extract features for {dataset.__len__()} items. ")

    audio_feat, visual_feat = torch.tensor([]), torch.tensor([])

    for i, batch in enumerate(dataset.batches()):
        frame, audio = batch["video"], batch["audio"]
        with torch.no_grad():  # Forward pass to get the features
            audio_feat = model.audio_model(audio)
            visual_feat = model.video_model(frame)
        logger.info(
            "Extracted features. "
            f"Audio shape: {audio_feat.shape}, "
            f"visual shape: {visual_feat.shape}"
        )
    # Binarize resulting feature matrix
    # Use GPU for processing
    # Store binarized feature matrix to file


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
