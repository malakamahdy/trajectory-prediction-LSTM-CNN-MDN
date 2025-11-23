import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# Graph Attention Layer
# --------------------------
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * out_dim, 1)

    def forward(self, h):
        """
        h: [N, F] node features (we'll treat batch agents as nodes)
        """
        N = h.size(0)
        Wh = self.W(h)  # [N, out]

        # All pair combinations i,j
        Wh_i = Wh.repeat(1, N).view(N * N, -1)
        Wh_j = Wh.repeat(N, 1)
        a_input = torch.cat([Wh_i, Wh_j], dim=1)  # [N*N, 2*out]

        e = self.attn(a_input).view(N, N)         # [N, N]
        attention = F.softmax(e, dim=1)           # normalized over neighbors

        h_prime = torch.mm(attention, Wh)         # [N, out]
        return h_prime

# --------------------------
# MDN Decoder
# --------------------------
class MDNDecoder(nn.Module):
    def __init__(self, input_dim, n_gauss, future):
        super().__init__()
        self.future = future
        self.n_gauss = n_gauss

        # For each Gaussian:
        #   mu:    T * 2
        #   sigma: T * 2
        #   pi:    1
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

# --------------------------
# Full ST-GAT Model
# --------------------------
class STGAT(nn.Module):
    def __init__(self, history=8, future=12, n_gauss=3):
        super().__init__()

        self.history = history
        self.future = future
        self.n_gauss = n_gauss

        # (1) Bidirectional LSTM encoder
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=64,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_proj = nn.Linear(64 * 2, 64)   # project bi-LSTM -> 64-dim

        # (2) Graph Attention Network (simple, over batch nodes)
        self.gat = GATLayer(64, 64)

        # (3) CNN-based map encoder
        self.map_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()   # -> 32-dim
        )

        # Combine social (GAT) + map features
        self.comb_proj = nn.Linear(64 + 32, 64)

        # MDN decoder for multi-modal future
        self.decoder = MDNDecoder(64, n_gauss, future)

    def forward(self, past, maps):
        """
        past: [B, H, 2]
        maps: [B, 3, Hm, Wm]
        """
        # LSTM encoding
        lstm_out, (h, c) = self.lstm(past)   # h: [2, B, 64]
        h_fwd = h[0]
        h_bwd = h[1]
        h_cat = torch.cat([h_fwd, h_bwd], dim=1)  # [B, 128]
        h_enc = self.lstm_proj(h_cat)             # [B, 64]

        # Social GAT over batch agents
        gat_out = self.gat(h_enc)                 # [B, 64]

        # Map encoding CNN
        map_feat = self.map_encoder(maps)         # [B, 32]

        # Combine and decode MDN
        z = torch.cat([gat_out, map_feat], dim=1) # [B, 96]
        z = self.comb_proj(z)                     # [B, 64]

        mu, sigma, pi = self.decoder(z)
        return mu, sigma, pi
