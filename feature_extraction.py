import logging
import os
from pathlib import Path
from time import time
import torch
from typing import Optional

from data_handling import VisXPData
from io_util import untar_input_file
from models import VisXPFeatureExtractionInput, Provenance
from nn_models import load_model_from_file


logger = logging.getLogger(__name__)


def apply_model(batch, model, device):
    frames, spectograms = batch["video"], batch["audio"]
    timestamps = batch["timestamp"].to(device)
    shots = batch["shot_boundaries"].to(device)
    # TODO: mask/disregard all zero frames/spectograms
    # (for the, now theoretical, case of only audio OR video existing)
    with torch.no_grad():  # Forward pass to get the features
        audio_feat = model.audio_model(spectograms)
        visual_feat = model.video_model(frames)
    result = torch.concat((timestamps.unsqueeze(1), shots, audio_feat, visual_feat), 1)
    return result


def run(
    feature_extraction_input: VisXPFeatureExtractionInput,
    model_base_mount: str,
    model_checkpoint_file: str,
    model_config_file: str,
    output_file_path: str,
) -> Optional[Provenance]:
    start_time = time()

    logger.info(f"Extracting features from: {feature_extraction_input.input_file_path}")

    # Step 1: set up GPU processing if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device is: {device}")

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

    # Step 4: Load spectograms + keyframes from file & preprocess
    dataset = VisXPData(
        datapath=Path(input_file_path),
        model_config_file=os.path.join(model_base_mount, model_config_file),
        device=device
    )

    # Step 5: Load model from file
    model = load_model_from_file(
        checkpoint_file=os.path.join(model_base_mount, model_checkpoint_file),
        config_file=os.path.join(model_base_mount, model_config_file),
        device=device,
    )
    # Switch model mode: in training mode, model layers behave differently!
    model.eval()

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
        parent_dir = str(Path(output_file_path).parent)
        logger.info(f"Checking if parent dir (source_id) exists: {parent_dir}")
        if not os.path.isdir(parent_dir):
            logger.info("Parent dir, did not exist, creating it now")
            os.makedirs(parent_dir)
        with open(output_file_path, "wb") as f:
            torch.save(obj=features, f=f)
            return True
    except Exception:
        logger.exception("Failed to save features to file")
    return False
