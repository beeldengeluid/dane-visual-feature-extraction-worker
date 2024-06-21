from data_handling import VisXPData
import torch
import pytest


@pytest.mark.parametrize("audio_too", [True, False])
def test_batches(audio_too):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = VisXPData(
        datapath="data/input-files/test_source_id",
        model_config_file=(
            "model/model_config.yml" if audio_too else "model/visualnet_config.yml"
        ),
        device=device,
        audio_too=audio_too,
    )
    for i, item in enumerate(dataset.batches(1)):
        index = int(item["timestamp"][0])

        for kind in ["video", "audio"]:
            if not audio_too and kind == "audio":
                continue
            this = item[kind][0]
            example = obtain_example(index, kind)
            assert torch.equal(example, this)


def obtain_example(i, kind):
    assert kind in ["video", "audio"]
    example_path = f"data/example_dataset/demo_{kind}_{i}.pt"
    with open(example_path, "rb") as f:
        example = torch.load(f)
    return example
