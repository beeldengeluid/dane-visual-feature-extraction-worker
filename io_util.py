import logging
import os
from pathlib import Path
import shutil
import tarfile
from time import time
from typing import Dict, List

from dane import Document
from dane.config import cfg
from dane.s3_util import S3Store, parse_s3_uri, validate_s3_uri
from models import (
    OutputType,
    Provenance,
    VisXPFeatureExtractionInput,
)


logger = logging.getLogger(__name__)
DANE_VISXP_PREP_TASK_KEY = "VISXP_PREP"
OUTPUT_FILE_BASE_NAME = "visxp_features"
INPUT_FILE_BASE_NAME = "visxp_prep"
TAR_GZ_EXTENSION = ".tar.gz"
S3_OUTPUT_TYPES: List[OutputType] = [
    OutputType.FEATURES,
    OutputType.PROVENANCE,
]  # only upload this output to S3


# make sure the necessary base dirs are there
def validate_data_dirs() -> bool:  # TODO: perhaps add model dir
    i_dir = Path(get_download_dir())
    o_dir = Path(get_base_output_dir())

    if not os.path.exists(i_dir.parent.absolute()):
        logger.info(
            f"{i_dir.parent.absolute()} does not exist. Make sure BASE_MOUNT_DIR exists before retrying"
        )
        return False

    # make sure the input and output dirs are there
    try:
        os.makedirs(i_dir, 0o755)
        logger.info("created VisXP input dir: {}".format(i_dir))
    except FileExistsError as e:
        logger.info(e)

    try:
        os.makedirs(o_dir, 0o755)
        logger.info("created VisXP output dir: {}".format(o_dir))
    except FileExistsError as e:
        logger.info(e)

    return True


# for each OutputType a subdir is created inside the base output dir
def generate_output_dirs(source_id: str) -> Dict[str, str]:
    base_output_dir = get_base_output_dir(source_id)
    output_dirs = {}
    for output_type in OutputType:
        output_dir = os.path.join(base_output_dir, output_type.value)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        output_dirs[output_type.value] = output_dir
    return output_dirs


# below this dir each processing module will put its output data in a subfolder
def get_base_output_dir(source_id: str = "") -> str:
    path_elements = [cfg.FILE_SYSTEM.BASE_MOUNT, cfg.FILE_SYSTEM.OUTPUT_DIR]
    if source_id:
        path_elements.append(source_id)
    return os.path.join(*path_elements)


# output file name of the final tar.gz that will be uploaded to S3
def get_archive_file_path(source_id: str) -> str:
    return os.path.join(
        get_base_output_dir(source_id),
        f"{OUTPUT_FILE_BASE_NAME}__{source_id}{TAR_GZ_EXTENSION}",
    )


def get_output_file_name(source_id: str, output_type: OutputType) -> str:
    output_file_name = ""
    match output_type:
        case OutputType.FEATURES:
            output_file_name = f"{OUTPUT_FILE_BASE_NAME}__{source_id}.pt"
        case OutputType.PROVENANCE:
            output_file_name = "provenance.json"
    return output_file_name


# output file name of the final .pt file that will be uploaded to S3
# TODO decide whether to tar.gz this as well
def get_output_file_path(source_id: str, output_type: OutputType) -> str:
    return os.path.join(
        get_base_output_dir(source_id),
        output_type.value,
        get_output_file_name(source_id, output_type),
    )


# e.g. s3://<bucket>/assets/<source_id>
def get_s3_base_uri(source_id: str) -> str:
    return f"s3://{os.path.join(cfg.OUTPUT.S3_BUCKET, cfg.OUTPUT.S3_FOLDER_IN_BUCKET, source_id)}"


# e.g. s3://<bucket>/assets/<source_id>/visxp_features__<source_id>.pt
# TODO adapt this
def get_s3_output_file_uri(source_id: str, output_type: OutputType) -> str:
    return (
        f"{get_s3_base_uri(source_id)}/{get_output_file_name(source_id, output_type)}"
    )


# NOTE: only use for test run & unit test with input that points to tar file!
# e.g. ./data/input-files/visxp_prep__testob.tar.gz
def get_source_id_from_tar(input_path: str) -> str:
    fn = os.path.basename(input_path)
    tmp = fn.split("__")
    source_id = tmp[1][: -len(TAR_GZ_EXTENSION)]
    logger.info(f"Using source_id: {source_id}")
    return source_id


# e.g. s3://<bucket>/assets/<source_id>/visxp_prep__<source_id>.tar.gz
# s3://your-bucket/assets/2101608170158176431__NOS_JOURNAAL_-WON01513227/visxp_prep__2101608170158176431__NOS_JOURNAAL_-WON01513227.tar.gz
def source_id_from_s3_uri(s3_uri: str) -> str:
    fn = os.path.basename(
        s3_uri
    )  # e.g. visxp_prep__2101608170158176431__NOS_JOURNAAL_-WON01513227.tar.gz
    fn = fn.replace(".tar.gz", "")
    source_id = "__".join(fn.split("__")[1:])  # remove the "visxp_prep__" bit
    return source_id


def delete_local_output(source_id: str) -> bool:
    output_dir = get_base_output_dir(source_id)
    logger.info(f"Deleting output folder: {output_dir}")
    if output_dir == os.sep or output_dir == ".":
        logger.warning(f"Rejected deletion of: {output_dir}")
        return False

    if not _is_valid_visxp_output(output_dir):
        logger.warning(
            f"Tried to delete a dir that did not contain VisXP output: {output_dir}"
        )
        return False

    try:
        shutil.rmtree(output_dir)
        logger.info(f"Cleaned up folder {output_dir}")
    except Exception:
        logger.exception(f"Failed to delete output dir {output_dir}")
        return False
    return True


# TODO implement some more, now checks presence of provenance dir
def _is_valid_visxp_output(output_dir: str) -> bool:
    return os.path.exists(os.path.join(output_dir, OutputType.PROVENANCE.value))


def _validate_transfer_config() -> bool:
    if any(
        [
            not x
            for x in [
                cfg.OUTPUT.S3_ENDPOINT_URL,
                cfg.OUTPUT.S3_BUCKET,
                cfg.OUTPUT.S3_FOLDER_IN_BUCKET,
            ]
        ]
    ):
        logger.warning(
            "TRANSFER_ON_COMPLETION configured without all the necessary S3 settings"
        )
        return False
    return True


# compresses all desired output dirs into a single tar and uploads it to S3
def transfer_output(source_id: str) -> bool:
    output_dir = get_base_output_dir(source_id)
    logger.info(f"Transferring {output_dir} to S3 (asset={source_id})")
    if not _validate_transfer_config():
        return False

    s3 = S3Store(cfg.OUTPUT.S3_ENDPOINT_URL)
    file_list = [os.path.join(output_dir, ot.value) for ot in S3_OUTPUT_TYPES]
    tar_file = get_archive_file_path(source_id)

    success = s3.transfer_to_s3(
        cfg.OUTPUT.S3_BUCKET,
        os.path.join(
            cfg.OUTPUT.S3_FOLDER_IN_BUCKET, source_id
        ),  # assets/<program ID>__<carrier ID>
        file_list,  # this list of subdirs will be compressed into the tar below
        tar_file,  # this file will be uploaded
    )
    if not success:
        logger.error(f"Failed to upload: {tar_file}")
        return False
    return True


def get_download_dir() -> str:
    return os.path.join(cfg.FILE_SYSTEM.BASE_MOUNT, cfg.FILE_SYSTEM.INPUT_DIR)


def get_base_input_dir(source_id: str) -> str:
    return os.path.join(get_download_dir(), source_id)


def delete_input_file(input_file: str, source_id: str, actually_delete: bool) -> bool:
    logger.info(f"Verifying deletion of input file: {input_file}")
    if actually_delete is False:
        logger.info("Configured to leave the input alone, skipping deletion")
        return True

    # first remove the input file
    try:
        os.remove(input_file)
        logger.info(f"Deleted VisXP input tar file: {input_file}")
    except OSError:
        logger.exception("Could not delete input file")
        return False

    # now remove the folders that were extracted from the input tar file
    base_input_dir = get_base_input_dir(source_id)
    try:
        for root, dirs, files in os.walk(base_input_dir):
            for d in dirs:
                dir_path = os.path.join(root, d)
                logger.info(f"Deleting {dir_path}")
                shutil.rmtree(dir_path)
        logger.info("Deleted extracted input dirs")
        os.removedirs(base_input_dir)
        logger.info(f"Finally deleted the base_input_dir: {base_input_dir}")
    except OSError:
        logger.exception("OSError while removing empty input file dirs")
    except FileNotFoundError:
        logger.exception("FileNotFoundError while removing empty input file dirs")

    return True  # return True even if empty dirs were not removed


def obtain_input_file(s3_uri: str) -> VisXPFeatureExtractionInput:

    if not validate_s3_uri(s3_uri):
        return VisXPFeatureExtractionInput(500, f"Invalid S3 URI: {s3_uri}")

    source_id = source_id_from_s3_uri(s3_uri)
    start_time = time()
    output_folder = get_base_input_dir(source_id)

    # TODO download the content into get_download_dir()
    s3 = S3Store(cfg.OUTPUT.S3_ENDPOINT_URL)
    bucket, object_name = parse_s3_uri(s3_uri)
    logger.info(f"OBJECT NAME: {object_name}")
    input_file_path = os.path.join(
        get_download_dir(),
        source_id,
        os.path.basename(object_name),  # i.e. visxp_prep__<source_id>.tar.gz
    )
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


def fetch_visxp_prep_s3_uri(handler, doc: Document) -> str:
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
