import torch
from torch.utils.data import DataLoader, random_split
from dataset import TrajectoryDataset
from baselines import LSTMBaseline
from utils import ADE, FDE
import os

# ANSI colors
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"

# Baseline save directory + path
BASELINE_DIR = "pretrained_baseline_model"
os.makedirs(BASELINE_DIR, exist_ok=True)
BASELINE_PATH = os.path.join(BASELINE_DIR, "lstm_baseline.pth")


def train_lstm_baseline(
    num_epochs=20,
    batch_size=32,
    history=8,
    future=12,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{CYAN}{BOLD}▶ Training LSTM baseline (no GAT, no MDN){RESET}")
    print(f"{CYAN}Device:{RESET} {device}")
    print(f"{CYAN}History steps (H):{RESET} {history}")
    print(f"{CYAN}Future steps (F):{RESET} {future}")
    print(f"{CYAN}Batch size:{RESET} {batch_size}")
    print(f"{CYAN}Epochs:{RESET} {num_epochs}")
    print()

    full_ds = TrajectoryDataset("data", history=history, future=future, verbose=False)
    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    print(f"{MAGENTA}Dataset info (LSTM baseline):{RESET}")
    print(f"  Total samples: {n_total}")
    print(f"  Train samples: {n_train}")
    print(f"  Val samples:   {n_val}")
    print()

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LSTMBaseline(history=history, future=future).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    print(f"{BOLD}Epoch |   Train MSE   |    Val MSE   |  Val ADE  |  Val FDE{RESET}")
    print("---------------------------------------------------------------")

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for past, fut, maps in train_dl:
            past = past.to(device)
            fut = fut.to(device)
            # maps is unused in this baseline

            optim.zero_grad()
            pred = model(past)
            loss = criterion(pred, fut)
            loss.backward()
            optim.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dl)

        model.eval()
        total_val_loss = 0.0
        ade_list, fde_list = [], []

        with torch.no_grad():
            for past, fut, maps in val_dl:
                past = past.to(device)
                fut = fut.to(device)

                pred = model(past)
                loss = criterion(pred, fut)
                total_val_loss += loss.item()

                ade_list.append(ADE(pred, fut).item())
                fde_list.append(FDE(pred, fut).item())

        avg_val_loss = total_val_loss / len(val_dl)
        avg_ade = sum(ade_list) / len(ade_list)
        avg_fde = sum(fde_list) / len(fde_list)

        print(
            f"{epoch:5d} | {avg_train_loss:11.4f} | {avg_val_loss:11.4f} |"
            f" {avg_ade:8.4f} | {avg_fde:8.4f}"
        )

    torch.save(model.state_dict(), BASELINE_PATH)
    print()
    print(f"{GREEN}◡̈ Saved trained LSTM baseline to {BASELINE_PATH}{RESET}")


if __name__ == "__main__":
    train_lstm_baseline()
