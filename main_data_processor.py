import logging
import os
from typing import Tuple, Optional

from dane.config import cfg
from dane.s3_util import validate_s3_uri
import feature_extraction
from io_util import (
    get_base_output_dir,
    get_output_file_path,
    get_s3_output_file_uri,
    generate_output_dirs,
    get_source_id_from_tar,
    obtain_input_file,
    transfer_output,
    delete_local_output,
    delete_input_file,
    validate_data_dirs,
)
from models import (
    CallbackResponse,
    VisXPFeatureExtractionOutput,
    VisXPFeatureExtractionInput,
    OutputType,
)
from nn_models import download_model_from_s3
from dane.provenance import (
    Provenance,
    obtain_software_versions,
    generate_initial_provenance,
    stop_timer_and_persist_provenance_chain,
)


logger = logging.getLogger(__name__)
DANE_WORKER_ID = "dane-visual-feature-extraction-worker"


# triggered by running: python worker.py --run-test-file
def run(input_file_path: str) -> Tuple[CallbackResponse, Optional[Provenance]]:
    # there must be an input file
    if not input_file_path:
        logger.error("input file empty")
        return {"state": 403, "message": "Error, no input file"}, []

    # check if the file system is setup properly
    if not validate_data_dirs():
        logger.info("ERROR: data dirs not configured properly")
        return {"state": 500, "message": "Input & output dirs not ok"}, []

    # create the top-level provenance
    top_level_provenance = generate_initial_provenance(
        name="VisXP feature extraction",
        description=(
            "Based on keyframes and corresponing audio spectograms, "
            "extract features by applying forward pass of a model"
        ),
        input_data={"input_file_path": input_file_path},
        parameters=dict(cfg.VISXP_EXTRACT),
        software_version=obtain_software_versions(DANE_WORKER_ID),
    )
    provenance_chain = []  # will contain the steps of the top-level provenance

    # S3 URI, local tar.gz or locally extracted tar.gz is allowed
    if validate_s3_uri(input_file_path):
        feature_extraction_input = obtain_input_file(input_file_path)
    else:
        if input_file_path.find(".tar.gz") != -1:
            source_id = get_source_id_from_tar(input_file_path)
        else:
            source_id = input_file_path.split("/")[-1]

        feature_extraction_input = VisXPFeatureExtractionInput(
            200,
            f"Processing tar.gz archive: {input_file_path}",
            source_id,
            input_file_path,
            None,  # no download provenance when using local file
        )

    # add the download provenance
    if feature_extraction_input.provenance:
        provenance_chain.append(feature_extraction_input.provenance)

    # first generate the output dirs
    generate_output_dirs(feature_extraction_input.source_id)

    # apply model to input & extract features
    proc_result = extract_visual_features(feature_extraction_input)

    if proc_result.provenance:
        provenance_chain.append(proc_result.provenance)

    # as a last piece of output, generate the provenance.json before packaging and uploading
    full_provenance_chain = stop_timer_and_persist_provenance_chain(
        provenance=top_level_provenance,
        output_data={
            "output_path": get_base_output_dir(feature_extraction_input.source_id),
            "output_uri": get_s3_output_file_uri(feature_extraction_input.source_id),
        },
        provenance_chain=provenance_chain,
        provenance_file_path=get_output_file_path(
            feature_extraction_input.source_id, OutputType.PROVENANCE
        ),
    )

    # if all is ok, apply the I/O steps on the outputted features
    validated_output: CallbackResponse = apply_desired_io_on_output(
        feature_extraction_input,
        proc_result,
        cfg.INPUT.DELETE_ON_COMPLETION,
        cfg.OUTPUT.DELETE_ON_COMPLETION,
        cfg.OUTPUT.TRANSFER_ON_COMPLETION,
    )
    logger.info("Results after applying desired I/O")
    logger.info(validated_output)
    return validated_output, full_provenance_chain


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
    if not all(
        key in cfg.INPUT
        for key in ["MODEL_CHECKPOINT_S3_URI", "MODEL_CONFIG_S3_URI", "S3_ENDPOINT_URL"]
    ):
        logger.error(
            "Incomplete config for downloading models from S3, please configure: INPUT.S3_ENDPOINT_URL, INPUT.MODEL_CONFIG_S3_URI, INPUT.MODEL_CHECKPOINT_S3_URI"
        )
        return False

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

    return VisXPFeatureExtractionOutput(
        200,
        "Succesfully extracted features",
        get_base_output_dir(feature_extraction_input.source_id),
        feature_extraction_provenance,
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
            "message": "Generated VisXP features, but could not delete the input file",
        }

    return {
        "state": 200,
        "message": "Successfully generated VisXP features to be used for similarity search",
    }
