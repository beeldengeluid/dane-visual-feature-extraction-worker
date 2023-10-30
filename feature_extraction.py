import logging
from time import time

from nn_models import load_model_from_file
import sys
import torch
import os
from pathlib import Path
from data_handling import VisXPData
from models import VisXPFeatureExtractionOutput
from provenance import generate_full_provenance_chain
from output_util import get_source_id, export_features

logger = logging.getLogger(__name__)


def extract_features(
    input_path: str, model_path: str, model_config_file: str, output_path: str
) -> VisXPFeatureExtractionOutput:
    start_time = time()
    # Step 0: this is the "processing ID" if you will
    source_id = get_source_id(input_path)
    logger.info(f"Extracting features for: {source_id}.")

    # Load spectograms + keyframes from file & preprocess
    dataset = VisXPData(Path(input_path), model_config_file=model_config_file)

    # Load model from file
    model = load_model_from_file(
        checkpoint_file=model_path,
        config_file=model_config_file,
    )
    # Switch model mode: in training mode, model layers behave differently!
    model.eval()

    # Apply model to data
    logger.info(f"Going to extract features for {dataset.__len__()} items. ")

    result_list = []
    for i, batch in enumerate(dataset.batches(batch_size=1)):
        frames, spectograms = batch["video"], batch["audio"]
        timestamps, shots = batch["timestamp"], batch["shot_boundaries"]
        with torch.no_grad():  # Forward pass to get the features
            audio_feat = model.audio_model(spectograms)
            visual_feat = model.video_model(frames)
        batch_result = torch.concat(
            (timestamps.unsqueeze(1), shots, audio_feat, visual_feat), 1
        )
        result_list.append(batch_result)
    result = torch.cat(result_list)
    destination = os.path.join(output_path, f"{source_id}.pt")
    export_features(result, destination=destination)
    provenance = generate_full_provenance_chain(
        start_time=start_time,
        input_path=input_path,
        provenance_chain=[],
        output_path=destination,
    )
    return VisXPFeatureExtractionOutput(
        200, "Succesfully extracted features", provenance
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

    extract_features(
        input_path="data/visxp_prep/ZQWO_DYnq5Q_000000",
        output_path="data/visxp_features",
        model_config_file="models/model_config.yml",
        model_path="models/checkpoint.tar",
    )


def example_function():
    return 0 == 0
