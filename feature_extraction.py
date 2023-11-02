import logging
from nn_models import load_model_from_file
import os
from pathlib import Path
from time import time
import torch

from data_handling import VisXPData
from io_util import (
    save_features_to_file,
    untar_input_file,
    get_output_file_name,
)
from models import VisXPFeatureExtractionOutput, VisXPFeatureExtractionInput
from provenance import generate_full_provenance_chain

logger = logging.getLogger(__name__)


def apply_model(batch, model, device):
    frames, spectograms = batch["video"], batch["audio"]
    timestamps = batch["timestamp"].to(device)
    shots = batch["shot_boundaries"].to(device)
    with torch.no_grad():  # Forward pass to get the features
        audio_feat = model.audio_model(spectograms)
        visual_feat = model.video_model(frames)
    result = torch.concat((timestamps.unsqueeze(1), shots, audio_feat, visual_feat), 1)
    return result


def extract_features(
    feature_extraction_input: VisXPFeatureExtractionInput,
    model_path: str,
    model_config_file: str,
    output_path: str,
) -> VisXPFeatureExtractionOutput:
    start_time = time()

    logger.warning(f"Extracting features into {output_path}")

    # Step 1: set up GPU processing if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device is: {device}")

    input_file_path = feature_extraction_input.input_file_path
    source_id = feature_extraction_input.source_id

    # Step 1: this is the "processing ID" if you will
    logger.info(f"Extracting features for: {source_id}.")

    # Step 2: check the type of input (tar.gz vs a directory)
    if input_file_path.find(".tar.gz") != -1:
        logger.info("Input is an archive, uncompressing it")
        untar_input_file(input_file_path)  # extracts contents in same dir
        input_file_path = str(
            Path(input_file_path).parent
        )  # change the input path to the parent dir
        logger.info(f"Changed input_file_path to: {input_file_path}")

    # Step 3: Load spectograms + keyframes from file & preprocess
    dataset = VisXPData(
        Path(input_file_path), model_config_file=model_config_file, device=device
    )

    # Step 4: Load model from file
    model = load_model_from_file(
        checkpoint_file=model_path,
        config_file=model_config_file,
        device=device,
    )
    # Switch model mode: in training mode, model layers behave differently!
    model.eval()

    # Step 5: Apply model to data
    logger.info(f"Going to extract features for {dataset.__len__()} items. ")

    result_list = []
    for i, batch in enumerate(dataset.batches(batch_size=256)):
        batch_result = apply_model(batch=batch, model=model, device=device)
        result_list.append(batch_result)

    # concatenate results and save to file
    result = torch.cat(result_list)
    destination = os.path.join(output_path, get_output_file_name(source_id))
    file_saved = save_features_to_file(result, destination=destination)

    if not file_saved:
        return VisXPFeatureExtractionOutput(
            500,
            f"Could not save extracted features to {destination}",
            destination,
            None,
        )

    # generate provenance, since all went well
    provenance = generate_full_provenance_chain(
        start_time=start_time,
        input_path=input_file_path,
        provenance_chain=[],
        output_path=destination,
    )
    return VisXPFeatureExtractionOutput(
        200, "Succesfully extracted features", destination, provenance
    )

    # Binarize resulting feature matrix
    # Use GPU for processing
    # Store binarized feature matrix to file
