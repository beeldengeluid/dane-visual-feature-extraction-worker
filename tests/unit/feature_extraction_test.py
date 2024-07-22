import os
import torch

import feature_extraction
from io_util import get_output_file_path
from models import OutputType, VisXPFeatureExtractionInput
import nn_models
import numpy as np
import pytest

UNIT_TEST_SOURCE_ID = "test_source_id"
UNIT_TEST_INPUT_PATH = f"./data/input-files/{UNIT_TEST_SOURCE_ID}"


@pytest.fixture
def load_model(request):
    type = request.param
    if type == "AVNet":
        checkpoint_file = "model/checkpoint.tar"
        config_file = "model/model_config.yml"
    else:
        checkpoint_file = "model/visualnet_checkpoint.tar"
        config_file = "model/visualnet_config.yml"

    model = nn_models.load_model_from_file(
        checkpoint_file=checkpoint_file, config_file=config_file, device="cpu"
    )
    yield (model, config_file)


@pytest.mark.parametrize("load_model", ["Visualnet"], indirect=True)
def test_extract_features(load_model):
    feature_file = get_output_file_path(UNIT_TEST_SOURCE_ID, OutputType.FEATURES)
    model, config_file = load_model
    print(f"FEATURE FILE: {feature_file}")
    feature_extraction.run(
        feature_extraction_input=VisXPFeatureExtractionInput(
            200,
            f"Thank you for unit testing: let's process {UNIT_TEST_INPUT_PATH}",
            UNIT_TEST_SOURCE_ID,
            UNIT_TEST_INPUT_PATH,
            None,  # no provenance needed in test
        ),
        model=model,
        device="cpu",
        model_config_path=config_file,
        output_file_path=feature_file,
    )

    features = torch.Tensor(np.load(feature_file))
    with open("./data/demo_concat_feat.pt", "rb") as f:
        example_features = torch.load(f)

    # make sure that we're comparing the proper vectors
    assert torch.equal(features[:, 0], example_features[:, 0])

    features = features[:, 3:]  # columns 0,1,2 hold timestamps & shot boundaries
    example_features = example_features[:, 513:]  # only visual part
    assert torch.all(torch.isclose(features, example_features))

    os.remove(feature_file)


@pytest.mark.legacy
@pytest.mark.parametrize("load_model", ["AVNet"], indirect=True)
def test_extract_features_legacy(load_model):
    feature_file = get_output_file_path(UNIT_TEST_SOURCE_ID, OutputType.FEATURES)
    model, config_file = load_model
    print(f"FEATURE FILE: {feature_file}")
    feature_extraction.run(
        feature_extraction_input=VisXPFeatureExtractionInput(
            200,
            f"Thank you for unit testing: let's process {UNIT_TEST_INPUT_PATH}",
            UNIT_TEST_SOURCE_ID,
            UNIT_TEST_INPUT_PATH,
            None,  # no provenance needed in test
        ),
        model=model,
        device="cpu",
        model_config_path=config_file,
        output_file_path=feature_file,
    )

    features = torch.Tensor(np.load(feature_file))
    with open("./data/demo_concat_feat.pt", "rb") as f:
        example_features = torch.load(f)

    # make sure that we're comparing the proper vectors
    assert torch.equal(features[:, 0], example_features[:, 0])

    features = features[:, 3:]  # columns 0,1,2 hold timestamps & shot boundaries
    example_features = example_features[:, 1:]  # column 0 holds timestamps/indices
    assert torch.all(torch.isclose(features, example_features))

    os.remove(feature_file)


def test_dummy():
    assert 1 == 1
