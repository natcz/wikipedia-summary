import torch
import torch.nn as nn

class LSTMSummarizer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMSummarizer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size // 2, batch_first=True)
        self.fc = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        summarization = self.fc(lstm_out[:, -1, :])
        return summarization