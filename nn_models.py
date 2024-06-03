import torch
from torch import nn
from yacs.config import CfgNode as CN
import logging
from typing import Optional

from dane.s3_util import validate_s3_uri, download_s3_uri


logger = logging.getLogger(__name__)


class VisualNet(nn.Module):
    def __init__(self, double_convolution=True):
        super(VisualNet, self).__init__()

        self.double_convolution = double_convolution

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(
            64, eps=0.001, momentum=0.99
        )  # Eps and momentum from keras default
        if self.double_convolution:
            self.conv1B = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
            self.conv1B_bn = nn.BatchNorm2d(
                64, eps=0.001, momentum=0.99
            )  # Eps and momentum from keras default

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(
            128, eps=0.001, momentum=0.99
        )  # Eps and momentum from keras default
        if self.double_convolution:
            self.conv2B = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
            self.conv2B_bn = nn.BatchNorm2d(
                128, eps=0.001, momentum=0.99
            )  # Eps and momentum from keras default

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(
            256, eps=0.001, momentum=0.99
        )  # Eps and momentum from keras default
        if self.double_convolution:
            self.conv3B = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.conv3B_bn = nn.BatchNorm2d(
                256, eps=0.001, momentum=0.99
            )  # Eps and momentum from keras default

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(
            512, eps=0.001, momentum=0.99
        )  # Eps and momentum from keras default
        if self.double_convolution:
            self.conv4B = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.conv4B_bn = nn.BatchNorm2d(
                512, eps=0.001, momentum=0.99
            )  # Eps and momentum from keras default
        self.maxpool_4 = nn.MaxPool2d(kernel_size=(14, 14), stride=None)

        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.relu = nn.GELU()

    def forward(self, x):
        # 1st conv block
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.relu(x)

        if self.double_convolution:
            x = self.conv1B(x)
            x = self.conv1B_bn(x)
            x = self.relu(x)

        x = self.maxpool(x)

        # 2nd conv block
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.relu(x)

        if self.double_convolution:
            x = self.conv2B(x)
            x = self.conv2B_bn(x)
            x = self.relu(x)

        x = self.maxpool(x)

        # 3rd conv block
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = self.relu(x)

        if self.double_convolution:
            x = self.conv3B(x)
            x = self.conv3B_bn(x)
            x = self.relu(x)

        x = self.maxpool(x)

        # 4th conv block
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = self.relu(x)

        if self.double_convolution:
            x = self.conv4B(x)
            x = self.conv4B_bn(x)
            x = self.relu(x)

        x = self.maxpool_4(x)

        return torch.flatten(x, 1)


class AudioNet(nn.Module):
    def __init__(self, double_convolution=True):
        super(AudioNet, self).__init__()

        self.double_convolution = double_convolution

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(
            64, eps=0.001, momentum=0.99
        )  # Eps and momentum from keras default
        if self.double_convolution:
            self.conv1B = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
            self.conv1B_bn = nn.BatchNorm2d(
                64, eps=0.001, momentum=0.99
            )  # Eps and momentum from keras default

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(
            128, eps=0.001, momentum=0.99
        )  # Eps and momentum from keras default
        if self.double_convolution:
            self.conv2B = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
            self.conv2B_bn = nn.BatchNorm2d(
                128, eps=0.001, momentum=0.99
            )  # Eps and momentum from keras default

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(
            256, eps=0.001, momentum=0.99
        )  # Eps and momentum from keras default
        if self.double_convolution:
            self.conv3B = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.conv3B_bn = nn.BatchNorm2d(
                256, eps=0.001, momentum=0.99
            )  # Eps and momentum from keras default

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(
            512, eps=0.001, momentum=0.99
        )  # Eps and momentum from keras default
        if self.double_convolution:
            self.conv4B = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.conv4B_bn = nn.BatchNorm2d(
                512, eps=0.001, momentum=0.99
            )  # Eps and momentum from keras default
        self.maxpool_4 = nn.MaxPool2d(kernel_size=(16, 6), stride=None)

        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.relu = nn.GELU()

    def forward(self, x):
        # 1st conv block
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.relu(x)

        if self.double_convolution:
            x = self.conv1B(x)
            x = self.conv1B_bn(x)
            x = self.relu(x)

        x = self.maxpool(x)

        # 2nd conv block
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.relu(x)

        if self.double_convolution:
            x = self.conv2B(x)
            x = self.conv2B_bn(x)
            x = self.relu(x)

        x = self.maxpool(x)

        # 3rd conv block
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = self.relu(x)

        if self.double_convolution:
            x = self.conv3B(x)
            x = self.conv3B_bn(x)
            x = self.relu(x)

        x = self.maxpool(x)

        # 4th conv block
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = self.relu(x)

        if self.double_convolution:
            x = self.conv4B(x)
            x = self.conv4B_bn(x)
            x = self.relu(x)

        x = self.maxpool_4(x)

        return torch.flatten(x, 1)


class AVNet(nn.Module):
    def __init__(self, num_classes, double_convolution=True):

        super(AVNet, self).__init__()

        self.video_model = VisualNet(double_convolution)
        self.video_model.fc = nn.Linear(512, num_classes)
        self.audio_model = AudioNet(double_convolution)
        self.audio_model.fc = nn.Linear(512, num_classes)
        self.lin1 = nn.Linear(1024, 512)
        self.relu = nn.GELU()
        self.lin2 = nn.Linear(512, num_classes)

    def forward(self, audio, video):

        video_out = self.video_model(video)
        audio_out = self.audio_model(audio)
        x = torch.cat((video_out, audio_out), dim=1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)

        return x


def download_model_from_s3(
    model_base_mount: str,
    model_checkpoint_s3_uri: str,
    model_config_s3_uri: str,
    s3_endpoint_url: Optional[str] = "",
) -> bool:
    logger.info("Trying to download models from S3")
    s3_uris = [model_checkpoint_s3_uri, model_config_s3_uri]

    if not all([validate_s3_uri(s3_uri) for s3_uri in s3_uris]):
        logger.error("Please provide valid S3 URI's only")
        return False

    for s3_uri in s3_uris:
        if not validate_s3_uri(s3_uri):
            logger.error(f"S3 URI invalid: {s3_uri}")
            return False
        success = download_s3_uri(s3_uri=s3_uri, output_folder=model_base_mount)
        if not success:
            logger.error(f"Could not download {s3_uri} into {model_base_mount}")
            return False
        logger.info(f"Downloaded {s3_uri} into {model_base_mount}")
    logger.info("Succesfully downloaded model and its config")
    return True


def load_model_from_file(checkpoint_file, config_file, device):
    logger.info(f"Loading {checkpoint_file} and using model config: {config_file}")
    checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
    with open(config_file, "r") as f:
        cfg = CN.load_cfg(f)
    if cfg.MODEL.TYPE == "AVNet":
        model = AVNet(
            num_classes=cfg.MODEL.N_CLASSES,
            double_convolution=cfg.MODEL.DOUBLE_CONVOLUTION,
        )
        model.load_state_dict(checkpoint["state_dict"])
    elif cfg.MODEL.TYPE == "VisualNet":
        model = VisualNet(double_convolution=cfg.MODEL.DOUBLE_CONVOLUTION)
        model.load_state_dict(checkpoint)
    else:
        logger.error(
            f"Unspupported model type ({cfg.MODEL.TYPE}) specified"
            f" in model config {config_file}"
        )

    model.to(device)
    # Switch model mode: in training mode, model layers behave differently!
    model.eval()
    return model


def convert_avnet_to_visualnet(
        av_path: str, av_config_path, v_path: str, v_config_path: str):
    '''Load model checkpoint for AV model from file.
       Obtain model parameters for Visualnet and store to file.
       Convert model config accordingly (strip off some elements)'''
    loaded_model = load_model_from_file(av_path, av_config_path, "cpu")
    state_dict = loaded_model.video_model.state_dict()
    # fc: extra linear layer added on top of separate A/V models for AV-net
    # never used in forward pass though
    # and raises Exception when loading model from file
    del state_dict['fc.weight']
    del state_dict['fc.bias']
    torch.save(state_dict, v_path)
    logger.info("Saved visualnet checkpoint to file")
    with open(av_config_path, "r") as f:
        cfg = CN.load_cfg(f)
    if True:
        cfg.MODEL = CN({
            'TYPE': 'VisualNet',
            'DOUBLE_CONVOLUTION': cfg.MODEL.DOUBLE_CONVOLUTION})
        cfg.INPUT = CN({
            'KEYFRAME': {
                'DIMENSIONALITY': cfg.INPUT.KEYFRAME.DIMENSIONALITY,
                'NORMALIZATION': cfg.INPUT.KEYFRAME.NORMALIZATION}})
    with open(v_config_path, 'w') as f:
        f.write(cfg.dump())
