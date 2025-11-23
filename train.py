import torch
from torch.utils.data import DataLoader, random_split
from dataset import TrajectoryDataset
from model import STGAT
from utils import mdn_loss
import os

# -------------------------------------------------------------
# ANSI color codes for nicer terminal output
# -------------------------------------------------------------
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"

# -------------------------------------------------------------
# Folder where ST-GAT models are saved
# -------------------------------------------------------------
MODEL_DIR = "pretrained_model"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "stgat_mdn.pth")


def train_model(
    num_epochs=20,
    batch_size=32,
    history=8,
    future=12,
    device=None,
    load_existing=False,
    model_path=MODEL_PATH
):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{CYAN}{BOLD}▶ Training ST-GAT + MDN model{RESET}")
    print(f"{CYAN}Device:{RESET} {device}")
    print(f"{CYAN}History steps (H):{RESET} {history}")
    print(f"{CYAN}Future steps (F):{RESET} {future}")
    print(f"{CYAN}Batch size:{RESET} {batch_size}")
    print(f"{CYAN}Epochs:{RESET} {num_epochs}")
    print()

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------
    full_ds = TrajectoryDataset("data", history=history, future=future)

    # ---------------------------------------------------------
    # FIXED TRAIN/VAL SPLIT (matches eval.py exactly)
    # ---------------------------------------------------------
    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train

    g = torch.Generator().manual_seed(42)   # ensure consistent split w/ eval
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    print(f"{MAGENTA}Dataset info:{RESET}")
    print(f"  Total samples: {n_total}")
    print(f"  Train samples: {n_train}")
    print(f"  Val samples:   {n_val}")
    print()

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ---------------------------------------------------------
    # Initialize model + optimizer
    # ---------------------------------------------------------
    model = STGAT(history=history, future=future, n_gauss=3).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ---------------------------------------------------------
    # Load existing weights (optional)
    # ---------------------------------------------------------
    if load_existing and os.path.exists(model_path):
        print(f"{GREEN}Loading existing weights from {model_path}...{RESET}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"{YELLOW}Starting NEW training session (random weights).{RESET}")
    print()

    print(f"{BOLD}Epoch |   Train Loss   |    Val Loss{RESET}")
    print("----------------------------------------")

    # ---------------------------------------------------------
    # Training Loop
    # ---------------------------------------------------------
    for epoch in range(num_epochs):
        model.train()
        total_train = 0.0

        for past, fut, maps in train_dl:
            past = past.to(device)
            fut = fut.to(device)
            maps = maps.to(device)

            optim.zero_grad()
            mu, sigma, pi = model(past, maps)
            loss = mdn_loss(mu, sigma, pi, fut)
            loss.backward()
            optim.step()

            total_train += loss.item()

        avg_train = total_train / len(train_dl)

        # --------------------- Validation ---------------------
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for past, fut, maps in val_dl:
                past = past.to(device)
                fut = fut.to(device)
                maps = maps.to(device)

                mu, sigma, pi = model(past, maps)
                loss = mdn_loss(mu, sigma, pi, fut)
                total_val += loss.item()

        avg_val = total_val / len(val_dl)

        print(f"{epoch:5d} | {avg_train:12.4f} | {avg_val:11.4f}")

    # ---------------------------------------------------------
    # Save final model checkpoint
    # ---------------------------------------------------------
    torch.save(model.state_dict(), model_path)
    print()
    print(f"{GREEN}◡̈ Saved trained ST-GAT model to {model_path}{RESET}")
