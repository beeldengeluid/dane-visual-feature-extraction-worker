import logging
import os
from time import time
import torch
from typing import List, Tuple, Optional

from dane import Document
from dane.config import cfg
from dane.s3_util import S3Store, parse_s3_uri, validate_s3_uri
from models import Provenance


logger = logging.getLogger(__name__)
DANE_VISXP_PREP_TASK_KEY = "VISXP_PREP"


# returns the basename of the input path
# throughout processing this is then used as a unique ID
# to keep track of the input/output
def get_source_id(input_path: str) -> str:
    fn = os.path.basename(input_path)
    return fn[0 : fn.rfind(".")] if "." in fn else fn


# below this dir each processing module will put its output data in a subfolder
def get_base_output_dir(source_id: str = "") -> str:
    path_elements = [cfg.FILE_SYSTEM.BASE_MOUNT, cfg.FILE_SYSTEM.OUTPUT_DIR]
    if source_id:
        path_elements.append(source_id)
    return os.path.join(*path_elements)


def export_features(features: torch.Tensor, destination: str):
    with open(os.path.join(destination), "wb") as f:
        torch.save(obj=features, f=f)


def delete_local_output(source_id: str) -> bool:
    # TODO: implement
    return True


def transfer_output(output_dir: str) -> bool:
    logger.info(f"Transferring {output_dir} to S3")
    # TODO: implement
    return True


def get_s3_base_url(source_id: str) -> str:
    return f"s3://{os.path.join(cfg.OUTPUT.S3_BUCKET, cfg.OUTPUT.S3_FOLDER_IN_BUCKET, source_id)}"


def obtain_files_to_upload_to_s3(output_dir: str) -> List[str]:
    s3_file_list = []
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            s3_file_list.append(os.path.join(root, f))
    return s3_file_list


# TODO: implement or replace function calls
def get_download_dir():
    return ""


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


def obtain_input_file(
    handler, doc: Document
) -> Tuple[Optional[str], Optional[Provenance]]:
    # first fetch and validate the obtained S3 URI
    s3_uri = _fetch_visxp_prep_s3_uri(handler, doc)
    if not validate_s3_uri(s3_uri):
        return None, None

    start_time = time()
    output_folder = get_download_dir()

    # TODO download the content into get_download_dir()
    s3 = S3Store(cfg.OUTPUT.S3_ENDPOINT_URL)
    bucket, object_name = parse_s3_uri(s3_uri)
    output_file = os.path.join(output_folder, os.path.basename(object_name))
    success = s3.download_file(bucket, object_name, output_folder)
    if success:
        download_provenance = Provenance(
            activity_name="download",
            activity_description="Download VISXP_PREP data",
            start_time_unix=start_time,
            processing_time_ms=time() - start_time,
            input_data={},
            output_data={"file_path": output_file},
        )
        return output_file, download_provenance
    logger.error("Failed to download VISXP_PREP data from S3")
    return None, None


def _fetch_visxp_prep_s3_uri(handler, doc: Document) -> str:
    logger.info("checking download worker output")
    possibles = handler.searchResult(doc._id, DANE_VISXP_PREP_TASK_KEY)
    logger.info(possibles)
    if len(possibles) > 0 and "s3_location" in possibles[0].payload:
        return possibles[0].payload.get("s3_location", "")
    logger.error("No s3_location found in VISXP_PREP result")
    return ""
