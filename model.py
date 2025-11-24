import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# MDN Decoder
# --------------------------
class MDNDecoder(nn.Module):
    def __init__(self, input_dim, n_gauss, future):
        super().__init__()
        self.future = future
        self.n_gauss = n_gauss

        output_dim = n_gauss * (future * 2 + future * 2 + 1)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, z):
        out = self.fc(z)

        mu_size = self.n_gauss * self.future * 2
        sigma_size = self.n_gauss * self.future * 2
        pi_size = self.n_gauss

        mu, sigma, pi = torch.split(out, [mu_size, sigma_size, pi_size], dim=1)

        mu = mu.view(-1, self.n_gauss, self.future, 2)          # [B,K,T,2]
        sigma = torch.exp(sigma.view(-1, self.n_gauss, self.future, 2))  # [B,K,T,2]
        pi = F.softmax(pi, dim=1)                               # [B,K]

        return mu, sigma, pi


# ------------------------------------------------------------
# Trajectory Prediction Model
# Bidirectional LSTM + CNN Map Encoder + MDN Decoder
# ------------------------------------------------------------
class LSTMCNNMDN(nn.Module):
    """
    Scene-Aware LSTM-MDN model for pedestrian trajectory prediction.
    Components:
        1) Bi-LSTM temporal encoder
        2) CNN map encoder for spatial context
        3) MDN decoder for multi-modal probabilistic forecasting
    """
    def __init__(self, history=8, future=12, n_gauss=3):
        super().__init__()

        self.history = history
        self.future = future
        self.n_gauss = n_gauss

        # (1) Bidirectional LSTM encoder for temporal dynamics
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=64,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_proj = nn.Linear(64 * 2, 64)   

        # (2) CNN-based map encoder for spatial context
        self.map_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()   # -> 32-dim
        )

        # Combine temporal + spatial features
        self.comb_proj = nn.Linear(64 + 32, 64)

        # (3) MDN decoder for multi-modal probabilistic predictions
        self.decoder = MDNDecoder(64, n_gauss, future)

    def forward(self, past, maps):
        """
        Predict future trajectories as a mixture of Gaussians.

        Args:
            past: [B, H, 2] - Historical positions (H timesteps)
            maps: [B, 3, Hm, Wm] - Rasterized scene maps

        Returns:
            mu:    [B, K, F, 2] - Means of Gaussian components
            sigma: [B, K, F, 2] - Standard deviations
            pi:    [B, K]       - Mixture weights
        """

        # (A) Encode temporal dynamics with bidirectional LSTM
        lstm_out, (h, c) = self.lstm(past)   # h: [2, B, 64]
        h_fwd = h[0]
        h_bwd = h[1]
        h_cat = torch.cat([h_fwd, h_bwd], dim=1)  # [B, 128]
        temporal_feat = self.lstm_proj(h_cat)     # [B, 64]

        # (B) Encode spatial context with CNN
        map_feat = self.map_encoder(maps)         # [B, 32]

        # (C) Combine temporal + spatial features
        z = torch.cat([temporal_feat, map_feat], dim=1) # [B, 96]
        z = self.comb_proj(z)                           # [B, 64]

        # (D) Decode to mixture of Gaussians
        mu, sigma, pi = self.decoder(z)
        return mu, sigma, pi
