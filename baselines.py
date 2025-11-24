import torch
import torch.nn as nn

# ------------------------------
# 1. Constant Velocity Baseline
# ------------------------------
def constant_velocity_predict(past, future_len):
    """
    past: [B,H,2]
    returns: [B,F,2]
    """
    # Estimate velocity from last two timesteps
    v = past[:, -1] - past[:, -2]   # [B,2]
    last = past[:, -1]              # [B,2]

    preds = []
    for k in range(1, future_len + 1):
        preds.append(last + k * v)
    preds = torch.stack(preds, dim=1)  # [B,F,2]
    return preds


# ------------------------------
# 2. Simple LSTM Baseline
# ------------------------------
class LSTMBaseline(nn.Module):
    """
    Simple LSTM that encodes past trajectory [H,2]
    and predicts a deterministic future [F,2] with no social interactions.
    """
    def __init__(self, history=8, future=12, hidden_size=64):
        super().__init__()
        self.future = future
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, future * 2)

    def forward(self, past):
        """
        past: [B,H,2]
        returns: [B,F,2]
        """
        _, (h, c) = self.lstm(past)   # h: [1,B,H]
        h = h.squeeze(0)              # [B,H]
        out = self.fc(h)              # [B,F*2]
        out = out.view(-1, self.future, 2)
        return out
