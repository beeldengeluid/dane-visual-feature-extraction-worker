import logging
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
from provenance import generate_full_provenance_chain


logger = logging.getLogger(__name__)


def extract_visual_features(
    feature_extraction_input: VisXPFeatureExtractionInput,
) -> VisXPFeatureExtractionOutput:
    logger.info("Starting VisXP visual feature extraction")
    start_time = time()
    feature_extraction_provenance = feature_extraction.run(
        feature_extraction_input,
        model_path=cfg.VISXP_EXTRACT.MODEL_PATH,
        model_config_file=cfg.VISXP_EXTRACT.MODEL_CONFIG_PATH,
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
