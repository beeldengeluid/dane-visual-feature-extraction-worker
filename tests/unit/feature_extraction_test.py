import os
import torch

import feature_extraction
from io_util import get_output_file_name, get_base_output_dir
from models import VisXPFeatureExtractionInput


UNIT_TEST_SOURCE_ID = "test_source_id"
UNIT_TEST_INPUT_PATH = f"./data/input-files/{UNIT_TEST_SOURCE_ID}"


def test_extract_features():
    feature_extraction.extract_features(
        feature_extraction_input=VisXPFeatureExtractionInput(
            200,
            f"Thank you for unit testing: let's process {UNIT_TEST_INPUT_PATH}",
            UNIT_TEST_SOURCE_ID,
            UNIT_TEST_INPUT_PATH,
            None,  # no provenance needed in test
        ),
        model_path="model/checkpoint.tar",
        model_config_file="model/model_config.yml",
        output_path=get_base_output_dir(UNIT_TEST_SOURCE_ID),
    )
    feature_file = f"./data/output-files/test_source_id/{get_output_file_name(UNIT_TEST_SOURCE_ID)}"
    with open(feature_file, "rb") as f:
        features = torch.load(f)
    with open("./data/demo_concat_feat.pt", "rb") as f:
        example_features = torch.load(f)

    # make sure that we're comparing the proper vectors
    assert torch.equal(features[:, 0], example_features[:, 0])

    features = features[:, 3:]  # columns 0,1,2 hold timestamps & shot boundaries
    example_features = example_features[:, 1:]  # column 0 holds timestamps/indices
    assert torch.equal(features, example_features)

    os.remove(feature_file)
