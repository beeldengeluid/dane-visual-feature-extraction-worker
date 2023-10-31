import feature_extraction
import os
import torch


def test_extract_features():
    feature_extraction.extract_features(
        input_path="tests/data",
        model_path="models/checkpoint.tar",
        model_config_file="models/model_config.yml",
        output_path="tests/data/",
    )
    feature_file = "tests/data/data.pt"
    with open(feature_file, "rb") as f:
        features = torch.load(f)
    with open("tests/data/demo_concat_feat.pt", "rb") as f:
        example_features = torch.load(f)

    # make sure that we're comparing the proper vectors
    assert torch.equal(features[:, 0], example_features[:, 0])

    features = features[:, 3:]  # columns 0,1,2 hold timestamps & shot boundaries
    example_features = example_features[:, 1:]  # column 0 holds timestamps/indices
    assert torch.equal(features, example_features)

    os.remove(feature_file)


def test_example_function():
    assert feature_extraction.example_function()
