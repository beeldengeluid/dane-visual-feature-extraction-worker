import logging
import os
from pathlib import Path
from time import time
import torch
from typing import Optional

from data_handling import VisXPData
from io_util import untar_input_file
from models import VisXPFeatureExtractionInput, Provenance
import numpy as np


logger = logging.getLogger(__name__)


def apply_model(batch, model, device):
    frames = batch["video"]
    timestamps = batch["timestamp"].to(device)
    shots = batch["shot_boundaries"].to(device)
    with torch.no_grad():  # Forward pass to get the features
        if "audio" in batch:
            spectrograms = batch["audio"]
            audio_feat = model.audio_model(spectrograms)
            visual_feat = model.video_model(frames)
            result = torch.concat(
                (timestamps.unsqueeze(1), shots, audio_feat, visual_feat), 1
            )
        else:
            visual_feat = model(frames)
            result = torch.concat((timestamps.unsqueeze(1), shots, visual_feat), 1)
    return result


def run(
    feature_extraction_input: VisXPFeatureExtractionInput,
    model: torch.nn.Module,
    device: torch.device,
    model_config_path: str,
    output_file_path: str,
    audio_too: bool = False,
) -> Optional[Provenance]:
    start_time = time()

    logger.info(f"Extracting features from: {feature_extraction_input.input_file_path}")

    # Step 2: verify the input file's existence
    input_file_path = feature_extraction_input.input_file_path
    if not os.path.exists(input_file_path):
        logger.error(f"Input path does not exist {input_file_path}")
        return None

    # This is the "processing ID" if you will
    source_id = feature_extraction_input.source_id
    logger.info(f"Extracting features for: {source_id}.")

    # Step 3: check the type of input (tar.gz vs a directory)
    if input_file_path.find(".tar.gz") != -1:
        logger.info("Input is an archive, uncompressing it")
        untar_input_file(input_file_path)  # extracts contents in same dir
        input_file_path = str(
            Path(input_file_path).parent
        )  # change the input path to the parent dir
        logger.info(f"Changed input_file_path to: {input_file_path}")

    # Step 4: Load spectrograms + keyframes from file & preprocess
    dataset = VisXPData(
        datapath=Path(input_file_path),
        model_config_file=model_config_path,
        device=device,
        audio_too=audio_too,
    )

    # Step 6: Apply model to data
    logger.info(f"Going to extract features for {dataset.__len__()} items. ")

    result_list = []
    for i, batch in enumerate(dataset.batches(batch_size=256)):
        batch_result = apply_model(batch=batch, model=model, device=device)
        result_list.append(batch_result)

    # Step 7: concatenate results and save to file
    result = torch.cat(result_list)
    file_saved = _save_features_to_file(result, output_file_path)

    if not file_saved:
        logger.error(f"Could not save extracted features to {output_file_path}")
        return None

    return Provenance(
        activity_name="VisXP feature extraction",
        activity_description=("Extract features vectors in .pt file"),
        start_time_unix=start_time,
        processing_time_ms=time() - start_time,
        input_data={"input_file_path": input_file_path},
        output_data={"output_file_path": output_file_path},
    )

    # Binarize resulting feature matrix
    # Use GPU for processing
    # Store binarized feature matrix to file


# saves the features to a local file, so it can be uploaded to S3
def _save_features_to_file(features: torch.Tensor, output_file_path: str) -> bool:
    logger.info(f"Saving features to {output_file_path}")
    try:
        features_np = np.array(features)
    except TypeError:
        # can't convert cuda:0 device type tensor to numpy.
        # Use Tensor.cpu() to copy the tensor to host memory first.
        np.array(features.cpu())
    try:
        parent_dir = str(Path(output_file_path).parent)
        logger.info(f"Checking if parent dir (source_id) exists: {parent_dir}")
        if not os.path.isdir(parent_dir):
            logger.info("Parent dir, did not exist, creating it now")
            os.makedirs(parent_dir)
        np.save(output_file_path, features_np)
        return True
    except Exception:
        logger.exception("Failed to save features to file")
        return False
