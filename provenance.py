import json
import logging
from time import time
from typing import List

from io_util import get_base_output_dir, get_output_file_path
from models import Provenance, OutputType


logger = logging.getLogger(__name__)
SOFTWARE_PROVENANCE_FILE = "/software_provenance.txt"
DANE_WORKER_ID = (
    "dane-visual-feature-extraction-worker"  # NOTE: should be same as GH repo name!
)
PROVENANCE_FILE = "provenance.json"


# Generates a the main Provenance object, which will embed/include the provided provenance_chain
def generate_full_provenance_chain(
    start_time: float,
    input_path: str,
    provenance_chain: List[Provenance],
    source_id: str,
) -> Provenance:
    provenance = Provenance(
        activity_name="VisXP feature extraction",
        activity_description=(
            "Based on keyframes and corresponing audio spectograms, "
            "extract features by applying forward pass of a model"
        ),
        start_time_unix=start_time,
        processing_time_ms=time() - start_time,
        parameters={},
        steps=provenance_chain,
        software_version=obtain_software_versions([DANE_WORKER_ID]),
        input_data={"input_path": input_path},
        output_data={"output_path": get_base_output_dir(source_id)},
    )

    prov_file = get_output_file_path(source_id, OutputType.PROVENANCE)
    with open(prov_file, "w", encoding="utf-8") as f:
        json.dump(provenance.to_json(), f, ensure_ascii=False, indent=4)
        logger.info(f"Wrote provenance info to file: {prov_file}")
    return provenance


# NOTE: software_provenance.txt is created while building the container image (see Dockerfile)
def obtain_software_versions(software_names):
    if isinstance(software_names, str):  # wrap a single software name in a list
        software_names = [software_names]
    try:
        with open(SOFTWARE_PROVENANCE_FILE) as f:
            urls = (
                {}
            )  # for some reason I couldnt manage a working comprehension for the below - SV
            for line in f.readlines():
                name, url = line.split(";")
                if name.strip() in software_names:
                    urls[name.strip()] = url.strip()
            assert len(urls) == len(software_names)
            return urls
    except FileNotFoundError:
        logger.info(
            f"Could not read {software_names} version"
            f"from file {SOFTWARE_PROVENANCE_FILE}: file does not exist"
        )
    except ValueError as e:
        logger.info(
            f"Could not parse {software_names} version"
            f"from file {SOFTWARE_PROVENANCE_FILE}. {e}"
        )
    except AssertionError:
        logger.info(
            f"Could not find {software_names} version"
            f"in file {SOFTWARE_PROVENANCE_FILE}"
        )
