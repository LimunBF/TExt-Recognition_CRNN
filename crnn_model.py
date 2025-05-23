import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, img_height, num_channels, num_classes):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x(H/2)x(W/2)

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128x(H/4)x(W/4)

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),  # 256x(H/8)x(W/4)

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),  # 512x(H/16)x(W/4)

            nn.Conv2d(512, 512, 2, 1, 0),  # valid conv
            nn.ReLU()
        )

        # Use a single LSTM with num_layers=2
        # Input size is 512 (from CNN output)
        # Output size from this LSTM will be 2 * 256 = 512 (due to bidirectional)
        self.rnn = nn.LSTM(512, 256, bidirectional=True, batch_first=True, num_layers=2)

        self.fc = nn.Linear(256 * 2, num_classes) # Output of final LSTM is 2*256 (bidirectional)

    def forward(self, x):
        # CNN
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)  # [b, c, w]
        conv = conv.permute(0, 2, 1)  # [b, w, c] = [batch, seq, features]

        # RNN
        rnn_out, _ = self.rnn(conv) # rnn_out now contains the output from the final LSTM layer

        # FC
        output = self.fc(rnn_out)
        return output  # [batch, seq_len, num_classes]