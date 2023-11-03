import logging
import os
from time import time

from dane.config import cfg
import feature_extraction
from io_util import (
    get_base_output_dir,
    get_output_file_path,
    transfer_output,
    delete_local_output,
    delete_input_file,
)
from models import (
    CallbackResponse,
    VisXPFeatureExtractionOutput,
    VisXPFeatureExtractionInput,
    OutputType,
)
from nn_models import download_model_from_s3
from provenance import generate_full_provenance_chain


logger = logging.getLogger(__name__)


# makes sure the models are available, if not downloads them from S3
def check_model_availability():
    logger.info("Checking if the model and its config are available")
    model_checkpoint_path = os.path.join(
        cfg.VISXP_EXTRACT.MODEL_BASE_MOUNT, cfg.VISXP_EXTRACT.MODEL_CHECKPOINT_FILE
    )
    model_config_path = os.path.join(
        cfg.VISXP_EXTRACT.MODEL_BASE_MOUNT, cfg.VISXP_EXTRACT.MODEL_CONFIG_FILE
    )
    if os.path.exists(model_checkpoint_path) and os.path.exists(model_config_path):
        logger.info("Models found, continuing")
        return True

    logger.info("Model not found, checking availability in S3")
    download_success = download_model_from_s3(
        cfg.VISXP_EXTRACT.MODEL_BASE_MOUNT,  # download models into this dir
        cfg.INPUT.MODEL_CHECKPOINT_S3_URI,  # model checkpoint file is stored here
        cfg.INPUT.MODEL_CONFIG_S3_URI,  # model config file is stored here
        cfg.INPUT.S3_ENDPOINT_URL,  # the endpoint URL of the S3 host
    )

    if not download_success:
        logger.error("Could not download models from S3")
        return False
    return True


def extract_visual_features(
    feature_extraction_input: VisXPFeatureExtractionInput,
) -> VisXPFeatureExtractionOutput:
    logger.info("Starting VisXP visual feature extraction")

    # first check if the model and its config are available
    if not check_model_availability():
        return VisXPFeatureExtractionOutput(500, "Could not find model and its config")

    start_time = time()  # skip counting the download of the model
    feature_extraction_provenance = feature_extraction.run(
        feature_extraction_input,
        model_base_mount=cfg.VISXP_EXTRACT.MODEL_BASE_MOUNT,
        model_checkpoint_file=cfg.VISXP_EXTRACT.MODEL_CHECKPOINT_FILE,
        model_config_file=cfg.VISXP_EXTRACT.MODEL_CONFIG_FILE,
        output_file_path=get_output_file_path(
            feature_extraction_input.source_id, OutputType.FEATURES
        ),
    )

    if not feature_extraction_provenance:
        return VisXPFeatureExtractionOutput(500, "Failed to extract features")

    # generate provenance, since all went well
    provenance = generate_full_provenance_chain(
        start_time=start_time,
        input_path=feature_extraction_input.input_file_path,
        provenance_chain=[],
        source_id=feature_extraction_input.source_id,
    )
    return VisXPFeatureExtractionOutput(
        200,
        "Succesfully extracted features",
        get_base_output_dir(feature_extraction_input.source_id),
        provenance,
    )


# assesses the output and makes sure input & output is handled properly
def apply_desired_io_on_output(
    feature_extraction_input: VisXPFeatureExtractionInput,
    proc_result: VisXPFeatureExtractionOutput,
    delete_input_on_completion: bool,
    delete_output_on_completetion: bool,
    transfer_output_on_completion: bool,
) -> CallbackResponse:
    # step 2: raise exception on failure
    if proc_result.state != 200:
        logger.error(f"Could not process the input properly: {proc_result.message}")
        # something went wrong inside the VisXP work processor, return that response here
        return {"state": proc_result.state, "message": proc_result.message}

    # step 3: process returned successfully, generate the output
    source_id = feature_extraction_input.source_id
    output_path = get_base_output_dir(source_id)  # TODO actually make sure this works

    # step 4: transfer the output to S3 (if configured so)
    transfer_success = True
    if transfer_output_on_completion:
        transfer_success = transfer_output(source_id)

    # failure of transfer, impedes the workflow, so return error
    if not transfer_success:
        return {
            "state": 500,
            "message": "Failed to transfer output to S3",
        }

    # clear the output files (if configured so)
    if delete_output_on_completetion:
        delete_success = delete_local_output(source_id)
        if not delete_success:
            # NOTE: just a warning for now, but one to keep an eye out for
            logger.warning(f"Could not delete output files: {output_path}")

    # step 8: clean the input file (if configured so)
    if not delete_input_file(
        feature_extraction_input.input_file_path,
        feature_extraction_input.source_id,
        delete_input_on_completion,
    ):
        return {
            "state": 500,
            "message": "Generated VISXP_PREP output, but could not delete the input file",
        }

    return {
        "state": 200,
        "message": "Successfully generated VisXP features to be used for similarity search",
    }
