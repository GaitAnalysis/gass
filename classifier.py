import torch
import torch.nn as nn
import torch.nn.functional as F


class GaitClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2):
        super(GaitClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        # input_shape (seq_len, input_size)
        # output_shape (num_classes)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size * 4)
        self.fc3 = nn.Linear(hidden_size * 4, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_classes)

        # print parameter count
        print(f"parameter count: {sum(p.numel() for p in self.parameters())}")

    def forward(self, x):
        # x shape (seq_len, input_size)
        # out shape (hidden_size)
        out, _ = self.lstm(x)
        out = self.fc1(out[-1, :])
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        return out
