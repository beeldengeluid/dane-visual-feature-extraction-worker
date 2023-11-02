import logging
import os
from pathlib import Path
import tarfile
from time import time
import torch

from dane import Document
from dane.config import cfg
from dane.s3_util import S3Store, parse_s3_uri, validate_s3_uri
from models import (
    CallbackResponse,
    Provenance,
    VisXPFeatureExtractionOutput,
    VisXPFeatureExtractionInput,
)


logger = logging.getLogger(__name__)
DANE_VISXP_PREP_TASK_KEY = "VISXP_PREP"
OUTPUT_FILE_BASE_NAME = "visxp_features"
INPUT_FILE_BASE_NAME = "visxp_prep"
INPUT_FILE_EXTENSION = ".tar.gz"


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

    if delete_input_on_completion:
        logger.warning("Deletion of input not supported yet")

    return {
        "state": 200,
        "message": "Successfully generated VisXP features to be used for similarity search",
    }


# NOTE: only use for test run & unit test with input that points to tar file!
# e.g. ./data/input-files/visxp_prep__testob.tar.gz
def get_source_id_from_tar(input_path: str) -> str:
    fn = os.path.basename(input_path)
    tmp = fn.split("__")
    source_id = tmp[1][:-len(INPUT_FILE_EXTENSION)]
    logger.info(f"Using source_id: {source_id}")
    return source_id


# below this dir each processing module will put its output data in a subfolder
def get_base_output_dir(source_id: str = "") -> str:
    path_elements = [cfg.FILE_SYSTEM.BASE_MOUNT, cfg.FILE_SYSTEM.OUTPUT_DIR]
    if source_id:
        path_elements.append(source_id)
    return os.path.join(*path_elements)


# output file name of the final .pt file that will be uploaded to S3
# TODO decide whether to tar.gz this as well
def get_output_file_name(source_id: str) -> str:
    return f"{OUTPUT_FILE_BASE_NAME}__{source_id}.pt"


# e.g. s3://<bucket>/assets/<source_id>
def get_s3_base_uri(source_id: str) -> str:
    return f"s3://{os.path.join(cfg.OUTPUT.S3_BUCKET, cfg.OUTPUT.S3_FOLDER_IN_BUCKET, source_id)}"


# e.g. s3://<bucket>/assets/<source_id>/visxp_features__<source_id>.pt
def get_s3_output_file_uri(source_id: str) -> str:
    return f"{get_s3_base_uri(source_id)}/{get_output_file_name(source_id)}"


# e.g. s3://<bucket>/assets/<source_id>/visxp_prep__<source_id>.tar.gz
# TODO add validation of 1st VisXP worker's S3 URI
def source_id_from_s3_uri(s3_uri: str) -> str:
    fn = os.path.basename(s3_uri)
    source_id = fn[: -len(".tar.gz")].split("__")[1]
    return f"{source_id}"


# saves the features to a local file, so it can be uploaded to S3
def save_features_to_file(features: torch.Tensor, destination: str) -> bool:
    try:
        with open(destination, "wb") as f:
            torch.save(obj=features, f=f)
            return True
    except Exception:
        logger.exception("Failed to save features to file")
    return False


def delete_local_output(source_id: str) -> bool:
    # TODO: implement
    return True


def transfer_output(output_dir: str) -> bool:
    logger.info(f"Transferring {output_dir} to S3")
    # TODO: implement
    return True


def get_download_dir():
    return os.path.join(cfg.FILE_SYSTEM.BASE_MOUNT, cfg.FILE_SYSTEM.INPUT_DIR)


# NOTE: untested
def delete_input_file(input_file: str, actually_delete: bool) -> bool:
    logger.info(f"Verifying deletion of input file: {input_file}")
    if actually_delete is False:
        logger.info("Configured to leave the input alone, skipping deletion")
        return True

    # first remove the input file
    try:
        os.remove(input_file)
        logger.info(f"Deleted VisXP input file: {input_file}")
    except OSError:
        logger.exception("Could not delete input file")
        return False

    # now remove the "chunked path" from /mnt/dane-fs/input-files/03/d2/8a/03d28a03643a981284b403b91b95f6048576c234/xyz.mp4
    try:
        os.chdir(get_download_dir())  # cd /mnt/dane-fs/input-files
        os.removedirs(
            f".{input_file[len(get_download_dir()):input_file.rfind(os.sep)]}"
        )  # /03/d2/8a/03d28a03643a981284b403b91b95f6048576c234
        logger.info("Deleted empty input dirs too")
    except OSError:
        logger.exception("OSError while removing empty input file dirs")
    except FileNotFoundError:
        logger.exception("FileNotFoundError while removing empty input file dirs")

    return True  # return True even if empty dirs were not removed


def obtain_input_file(handler, doc: Document) -> VisXPFeatureExtractionInput:
    # first fetch and validate the obtained S3 URI
    # TODO make sure this is a valid S3 URI
    s3_uri = _fetch_visxp_prep_s3_uri(handler, doc)
    if not validate_s3_uri(s3_uri):
        return VisXPFeatureExtractionInput(500, f"Invalid S3 URI: {s3_uri}")

    start_time = time()
    output_folder = get_download_dir()

    # TODO download the content into get_download_dir()
    s3 = S3Store(cfg.OUTPUT.S3_ENDPOINT_URL)
    bucket, object_name = parse_s3_uri(s3_uri)
    input_file_path = os.path.join(output_folder, os.path.basename(object_name))
    success = s3.download_file(bucket, object_name, output_folder)
    if success:
        # TODO uncompress the visxp_prep.tar.gz

        provenance = Provenance(
            activity_name="download",
            activity_description="Download VISXP_PREP data",
            start_time_unix=start_time,
            processing_time_ms=time() - start_time,
            input_data={},
            output_data={"file_path": input_file_path},
        )
        return VisXPFeatureExtractionInput(
            200,
            f"Failed to download: {s3_uri}",
            source_id_from_s3_uri(s3_uri),  # source_id
            input_file_path,  # locally downloaded .tar.gz
            provenance,
        )
    logger.error("Failed to download VISXP_PREP data from S3")
    return VisXPFeatureExtractionInput(500, f"Failed to download: {s3_uri}")


def _fetch_visxp_prep_s3_uri(handler, doc: Document) -> str:
    logger.info("checking download worker output")
    possibles = handler.searchResult(doc._id, DANE_VISXP_PREP_TASK_KEY)
    logger.info(possibles)
    if len(possibles) > 0 and "s3_location" in possibles[0].payload:
        return possibles[0].payload.get("s3_location", "")
    logger.error("No s3_location found in VISXP_PREP result")
    return ""


# untars visxp_prep__<source_id>.tar.gz into the same dir
def untar_input_file(tar_file_path: str):
    logger.info(f"Uncompressing {tar_file_path}")
    with tarfile.open(tar_file_path) as tar:
        tar.extractall(path=str(Path(tar_file_path).parent), filter="data")  # type: ignore
