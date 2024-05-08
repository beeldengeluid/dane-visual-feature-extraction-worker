from dataclasses import dataclass, field, fields, is_dataclass
from dane.provenance import Provenance
from enum import Enum
from typing import Optional, TypedDict


# returned by callback()
class CallbackResponse(TypedDict):
    state: int
    message: str


# These are the types of output this worker (possibly) provides (depending on configuration)
class OutputType(Enum):
    FEATURES = "features"  # produced by feature_extraction.py
    PROVENANCE = "provenance"  # produced by provenance.py


# NOTE https://stackoverflow.com/questions/20670732/is-input-a-keyword-in-python

def provenance_from_dict(input: dict) -> Optional[Provenance]:
    """Converts provenance from a dictionary into a Provenance object. Calls itself
    recursively to handle each part of the provenance object
    :param input - this should be the dict with the provenance on the first call
    :returns the provenance as a Provenance object"""   
    return Provenance(
        activity_name=input.get("activity_name", ""),
        activity_description=input.get("activity_description", ""),
        processing_time_ms=input.get("processing_time_ms", -1),
        start_time_unix=input.get("start_time_unix", -1),
        parameters=input.get("parameters", {}),
        software_version=input.get("software_version", {}),
        input_data=input.get("input_data", {}),
        output_data=input.get("output_data", {}),
        steps=[provenance_from_dict(step) for step in input.get("steps", [])]
    )


@dataclass
class VisXPFeatureExtractionInput:
    state: int  # HTTP status code
    message: str  # error/sucess message
    source_id: str = ""  # <program ID>__<carrier ID>
    input_file_path: str = ""  # where the visxp_prep.tar.gz was downloaded
    provenance: Optional[Provenance] = None  # mostly: how long did it take to download


@dataclass
class VisXPFeatureExtractionOutput:
    state: int  # HTTP status code
    message: str  # error/success message
    output_file_path: str = ""  # where to store the extracted features
    provenance: Optional[Provenance] = None  # feature extraction provenance
