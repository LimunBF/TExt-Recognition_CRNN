import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, img_height, num_channels, num_classes):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, 1, 1),  # -> [B,64,H,W]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                   # -> [B,64,H/2,W/2]

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                   # -> [B,128,H/4,W/4]

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),         # -> [B,256,H/8,W/4]

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),         # -> [B,512,H/16,W/4]

            nn.Conv2d(512, 512, kernel_size=(2, 1), stride=(1, 1)),  # -> [B,512,H/16 - 1,W/4]
            nn.ReLU(),
        )

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        conv = self.cnn(x)             # [B, C, H', W']
        b, c, h, w = conv.size()
        conv = conv.permute(0, 3, 1, 2)   # [B, W', C, H']
        conv = conv.contiguous().view(b, w, c * h)  # [B, W', C*H']

        rnn_out, _ = self.rnn(conv)     # [B, T, 512]
        output = self.fc(rnn_out)       # [B, T, num_classes]
        return output
