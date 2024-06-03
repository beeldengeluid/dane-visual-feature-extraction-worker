import os
import torch

import feature_extraction
from io_util import get_output_file_path
from models import OutputType, VisXPFeatureExtractionInput
import numpy as np
import pytest

UNIT_TEST_SOURCE_ID = "test_source_id"
UNIT_TEST_INPUT_PATH = f"./data/input-files/{UNIT_TEST_SOURCE_ID}"


def test_extract_features():
    feature_file = get_output_file_path(UNIT_TEST_SOURCE_ID, OutputType.FEATURES)
    print(f"FEATURE FILE: {feature_file}")
    feature_extraction.run(
        feature_extraction_input=VisXPFeatureExtractionInput(
            200,
            f"Thank you for unit testing: let's process {UNIT_TEST_INPUT_PATH}",
            UNIT_TEST_SOURCE_ID,
            UNIT_TEST_INPUT_PATH,
            None,  # no provenance needed in test
        ),
        model_base_mount="model",
        model_checkpoint_file="visualnet_checkpoint.tar",
        model_config_file="visualnet_config.yml",
        output_file_path=feature_file,
        audio_too=False,
    )

    features = torch.Tensor(np.load(feature_file))
    with open("./data/demo_concat_feat.pt", "rb") as f:
        example_features = torch.load(f)

    # make sure that we're comparing the proper vectors
    assert torch.equal(features[:, 0], example_features[:, 0])

    features = features[:, 3:]  # columns 0,1,2 hold timestamps & shot boundaries
    example_features = example_features[:, 513:]  # only visual part
    assert torch.equal(features, example_features)

    os.remove(feature_file)


@pytest.mark.legacy
def test_extract_features_legacy():
    feature_file = get_output_file_path(UNIT_TEST_SOURCE_ID, OutputType.FEATURES)
    print(f"FEATURE FILE: {feature_file}")
    feature_extraction.run(
        feature_extraction_input=VisXPFeatureExtractionInput(
            200,
            f"Thank you for unit testing: let's process {UNIT_TEST_INPUT_PATH}",
            UNIT_TEST_SOURCE_ID,
            UNIT_TEST_INPUT_PATH,
            None,  # no provenance needed in test
        ),
        model_base_mount="model",
        model_checkpoint_file="checkpoint.tar",
        model_config_file="model_config.yml",
        output_file_path=feature_file,
        audio_too=True,
    )

    features = torch.Tensor(np.load(feature_file))
    with open("./data/demo_concat_feat.pt", "rb") as f:
        example_features = torch.load(f)

    # make sure that we're comparing the proper vectors
    assert torch.equal(features[:, 0], example_features[:, 0])

    features = features[:, 3:]  # columns 0,1,2 hold timestamps & shot boundaries
    example_features = example_features[:, 1:]  # column 0 holds timestamps/indices
    assert torch.equal(features, example_features)

    os.remove(feature_file)


def test_dummy():
    assert 1 == 1
