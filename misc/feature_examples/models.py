import torch
import torch.nn as nn

class VisualNet(nn.Module):
    def __init__(self, double_convolution=True):
        super(VisualNet, self).__init__()

        self.double_convolution = double_convolution

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(64, eps=0.001, momentum=0.99)    #Eps and momentum from keras default
        if self.double_convolution:
          self.conv1B = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
          self.conv1B_bn = nn.BatchNorm2d(64, eps=0.001, momentum=0.99)    #Eps and momentum from keras default

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(128, eps=0.001, momentum=0.99)    #Eps and momentum from keras default
        if self.double_convolution:
          self.conv2B = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
          self.conv2B_bn = nn.BatchNorm2d(128, eps=0.001, momentum=0.99)    #Eps and momentum from keras default

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(256, eps=0.001, momentum=0.99)    #Eps and momentum from keras default
        if self.double_convolution:
          self.conv3B = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
          self.conv3B_bn = nn.BatchNorm2d(256, eps=0.001, momentum=0.99)    #Eps and momentum from keras default

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(512, eps=0.001, momentum=0.99)    #Eps and momentum from keras default
        if self.double_convolution:
          self.conv4B = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
          self.conv4B_bn = nn.BatchNorm2d(512, eps=0.001, momentum=0.99)    #Eps and momentum from keras default         
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
        self.conv1_bn = nn.BatchNorm2d(64, eps=0.001, momentum=0.99)    #Eps and momentum from keras default
        if self.double_convolution:
            self.conv1B = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
            self.conv1B_bn = nn.BatchNorm2d(64, eps=0.001, momentum=0.99)    #Eps and momentum from keras default

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(128, eps=0.001, momentum=0.99)    #Eps and momentum from keras default
        if self.double_convolution:
            self.conv2B = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
            self.conv2B_bn = nn.BatchNorm2d(128, eps=0.001, momentum=0.99)    #Eps and momentum from keras default

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(256, eps=0.001, momentum=0.99)    #Eps and momentum from keras default
        if self.double_convolution:
            self.conv3B = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.conv3B_bn = nn.BatchNorm2d(256, eps=0.001, momentum=0.99)    #Eps and momentum from keras default

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(512, eps=0.001, momentum=0.99)    #Eps and momentum from keras default
        if self.double_convolution:
            self.conv4B = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.conv4B_bn = nn.BatchNorm2d(512, eps=0.001, momentum=0.99)    #Eps and momentum from keras default         
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
    
class AVNet_multihead(nn.Module):
    
    def __init__(self, num_classes, double_convolution=True, n_heads=10, num_overclusters=100):

        super(AVNet_multihead, self).__init__()

        # Base models
        self.video_model = VisualNet(double_convolution)
        self.audio_model = AudioNet(double_convolution)
        self.lin1 = nn.Linear(1024, 512)
        self.relu = nn.GELU()

        # Heads, one for video, one for audio, and one for both

        self.video_model.heads = [nn.Linear(512, num_classes)]
        self.audio_model.heads = [nn.Linear(512, num_classes)]
        self.heads = [nn.Linear(512, num_classes)]
        if n_heads > 1:
          for i in range(n_heads-1):
            self.video_model.heads.append(nn.Linear(512, num_overclusters))
            self.audio_model.heads.append(nn.Linear(512, num_overclusters))
            self.heads.append(nn.Linear(512, num_overclusters))

    def forward(self, audio, video):

        video_out = self.video_model(video)
        audio_out = self.audio_model(audio)
        x = torch.cat((video_out, audio_out), dim=1)
        x = self.lin1(x)
        x = self.relu(x)

        outputs = [head.cuda()(x) for head in self.heads]

        return outputs