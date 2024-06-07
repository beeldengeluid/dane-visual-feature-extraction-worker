import pytest
import nn_models
import os
import shutil


TMP_DIR = "tmp"
AV_CHECKPOINT = f"{TMP_DIR}/av_checkpoint.pt"
AV_CONFIG = f"{TMP_DIR}/av_config.yml"
VISUAL_CHECKPOINT = f"{TMP_DIR}/vis_checkpoint.pt"
VISUAL_CONFIG = f"{TMP_DIR}/vis_config.yml"
NUM_CLASSES = 2
DOUBLE_CONVOLUTION = True


@pytest.fixture(scope="module")
def setup():
    if os.path.exists(TMP_DIR):
        print(f"TMP_DIR ({TMP_DIR}) exists, abort.")
        assert False
    else:
        os.makedirs(TMP_DIR)
    yield
    shutil.rmtree(TMP_DIR)


@pytest.fixture
def create_model_files(setup):
    # Create config files
    cfg = nn_models.CN(
        {"MODEL": {"TYPE": "VisualNet", "DOUBLE_CONVOLUTION": DOUBLE_CONVOLUTION}}
    )
    with open(VISUAL_CONFIG, "w") as f:
        f.write(cfg.dump())
    cfg["MODEL"]["TYPE"] = "AVNet"
    cfg["MODEL"]["N_CLASSES"] = NUM_CLASSES
    with open(AV_CONFIG, "w") as f:
        f.write(cfg.dump())
    # Create checkpoint files
    model = nn_models.AVNet(num_classes=2, double_convolution=True)
    # import pdb
    # assert False
    # model.apply(nn_models.nn.init.uniform_)
    nn_models.torch.save({"state_dict": model.state_dict()}, AV_CHECKPOINT)
    state_dict = model.video_model.state_dict()
    del state_dict["fc.weight"]
    del state_dict["fc.bias"]
    nn_models.torch.save({"state_dict": state_dict}, VISUAL_CHECKPOINT)
    yield


def test_load_model_from_file(create_model_files):
    for checkpoint, config_file in [
        (AV_CHECKPOINT, AV_CONFIG),
        (VISUAL_CHECKPOINT, VISUAL_CONFIG),
    ]:
        nn_models.load_model_from_file(
            checkpoint_file=checkpoint,
            config_file=config_file,
            device=nn_models.torch.device("cpu"),
        )
