import logging
import timeit
import torch
from feature_extraction import apply_model
from nn_models import load_model_from_file
import sys

logger = logging.getLogger(__name__)


def generate_random_batch(device, batch_size=256):
    batch = {
        'video': torch.randn(batch_size, 3, 256, 256).to(device),
        'audio': torch.randn(batch_size, 1, 257, 99).to(device),
        'timestamp': torch.randn(batch_size).to(device),
        'shot_boundaries': torch.randn(batch_size, 2).to(device),
    }
    return batch


def time_feature_extraction(model_path, model_config_file):
    if torch.cuda.is_available():
        devices = ["cuda:0", "cpu"]
    else:
        devices = ["cpu"]

    for device in devices:
        logger.info(f"Running for {device}")
        batch = generate_random_batch(device=device)
        model = load_model_from_file(
            checkpoint_file=model_path,
            config_file=model_config_file,
            device=device,
        )
        logger.info(f"Setup for {device} is done")
        t = timeit.Timer(lambda: apply_model(batch=batch, model=model, device=device))
        logger.info(f"The time taken for {device} is {t.timeit(5)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,  # configure a stream handler only for now (single handler)
        format="%(asctime)s|%(levelname)s|%(process)d|%(module)s"
        "|%(funcName)s|%(lineno)d|%(message)s",
    )

    time_feature_extraction(
        model_config_file="models/model_config.yml",
        model_path="models/checkpoint.tar",
    )
