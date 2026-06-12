import torch
import torch.nn as nn

class AudioLSTM(nn.Module):
    def __init__(self, input_features=13, hidden_size=64, num_layers=2, num_classes=2):
        super(AudioLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # The LSTM layer automatically handles its own zero-initialized hidden states.
        lstm_out, _ = self.lstm(x)
        
        # Get the output of the last time step.
        last_time_step_output = lstm_out[:, -1, :]
        
        # Pass it to the final layer.
        logits = self.fc(last_time_step_output)
        
        return logits