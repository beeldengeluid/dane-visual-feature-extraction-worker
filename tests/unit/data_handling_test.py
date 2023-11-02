from data_handling import VisXPData
import torch


def test_batches():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = VisXPData(
        "data/input-files/test_source_id",
        device=device,
        model_config_file="model/model_config.yml",
    )
    for i, item in enumerate(dataset.batches(1)):
        index = int(item["timestamp"][0])

        for kind in ["video", "audio"]:
            this = item[kind][0]
        example = obtain_example(index, kind)
        assert torch.equal(example, this)


def obtain_example(i, kind):
    assert kind in ["frame", "audio"]
    example_path = f"data/example_dataset/demo_{kind}_{i}.pt"
    with open(example_path, "rb") as f:
        example = torch.load(f)
    return example
