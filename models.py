from dataclasses import dataclass
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
@dataclass
class Provenance:
    activity_name: str
    activity_description: str
    start_time_unix: float
    processing_time_ms: float
    input_data: dict[str, str]
    output_data: dict[str, str]
    parameters: Optional[dict] = None
    software_version: Optional[dict[str, str]] = None
    steps: Optional[list["Provenance"]] = None  # a list of subactivity provenance items

    def to_json(self):
        return {
            "activity_name": self.activity_name,
            "activity_description": self.activity_description,
            "processing_time_ms": self.processing_time_ms,
            "start_time_unix": self.start_time_unix,
            "parameters": self.parameters,  # .to_json
            "software_version": self.software_version,  # .to_json
            "input_data": self.input_data,  # .to_json
            "output_data": self.output_data,  # .to_json
            "steps": [step.to_json for step in self.steps],
        }


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
