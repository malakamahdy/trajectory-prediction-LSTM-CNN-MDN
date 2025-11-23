import torch
from torch.utils.data import DataLoader, random_split
from dataset import TrajectoryDataset
from model import STGAT
from baselines import constant_velocity_predict, LSTMBaseline
from utils import ADE, FDE
import os
import numpy as np

# ANSI color codes for pretty terminal formatting :)
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"

# Pretrained model folder paths
MODEL_DIR = "pretrained_model"
BASELINE_DIR = "pretrained_baseline_model"
MODEL_PATH = os.path.join(MODEL_DIR, "stgat_mdn.pth")
BASELINE_PATH = os.path.join(BASELINE_DIR, "lstm_baseline.pth")


def mdn_best_mode(mu, pi):
    """
    Extract the most likely predicted trajectory from the MDN output.
    """
    best_idx = torch.argmax(pi, dim=1)
    B, K, T, _ = mu.shape
    idx = best_idx.view(B, 1, 1, 1).expand(-1, 1, T, 2)
    best_mu = torch.gather(mu, 1, idx).squeeze(1)
    return best_mu


def evaluate(device=None, batch_size=64, history=8, future=12):
    """
    Evaluate ST-GAT+MDN, Constant Velocity baseline, and LSTM baseline
    on a consistent 20% test split (fixed random seed).
    """

    # -------------------- 
    # Device 
    # --------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{MAGENTA}{BOLD}===============================================")
    print("             MODEL EVALUATION")
    print("===============================================")
    print(f"{RESET}")
    print(f"{CYAN}Device:{RESET} {device}")
    print(f"{CYAN}History steps (H):{RESET} {history}")
    print(f"{CYAN}Future steps (F):{RESET} {future}")
    print()

    # -------------------- 
    # Dataset & Test Split 
    #--------------------
    full_ds = TrajectoryDataset("data", history=history, future=future, verbose=False)
    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train

    # FIXED SEED, same split in train.py and eval.py
    g = torch.Generator().manual_seed(42)
    _, test_ds = random_split(full_ds, [n_train, n_test], generator=g)

    print(f"{MAGENTA}Dataset info (Evaluation):{RESET}")
    print(f"  Total samples: {n_total}")
    print(f"  Test samples:  {n_test}")
    print()

    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # -------------------- 
    # Load ST-GAT Model 
    # --------------------
    if not os.path.exists(MODEL_PATH):
        print(f"{YELLOW}⚠ ST-GAT model not found at {MODEL_PATH}. Train it with python main.py{RESET}")
        return

    stgat = STGAT(history=history, future=future, n_gauss=3).to(device)
    stgat.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    stgat.eval()
    print(f"{GREEN}Loaded ST-GAT+MDN model from {MODEL_PATH}{RESET}")

    # -------------------- 
    # Load LSTM Baseline 
    # --------------------
    lstm_baseline = LSTMBaseline(history=history, future=future).to(device)
    if os.path.exists(BASELINE_PATH):
        lstm_baseline.load_state_dict(torch.load(BASELINE_PATH, map_location=device))
        print(f"{GREEN}Loaded LSTM baseline from {BASELINE_PATH}{RESET}")
    else:
        print(f"{YELLOW}⚠ LSTM baseline not found at {BASELINE_PATH}. Evaluating untrained.{RESET}")
    lstm_baseline.eval()

    # Metric accumulators
    ade_stgat, fde_stgat = [], []
    ade_cv, fde_cv = [], []
    ade_lstm, fde_lstm = [], []

    # -------------------- 
    # Evaluation Loop 
    # --------------------
    with torch.no_grad():
        for past, fut, maps in test_dl:
            past = past.to(device)
            fut = fut.to(device)
            maps = maps.to(device)

            # ST-GAT+MDN
            mu, sigma, pi = stgat(past, maps)
            pred_stgat = mdn_best_mode(mu, pi)
            ade_stgat.append(ADE(pred_stgat, fut).item())
            fde_stgat.append(FDE(pred_stgat, fut).item())

            # Constant Velocity
            pred_cv = constant_velocity_predict(past, future)
            ade_cv.append(ADE(pred_cv, fut).item())
            fde_cv.append(FDE(pred_cv, fut).item())

            # LSTM Baseline
            pred_lstm = lstm_baseline(past)
            ade_lstm.append(ADE(pred_lstm, fut).item())
            fde_lstm.append(FDE(pred_lstm, fut).item())

    # -------------------- 
    # Averages 
    # --------------------
    models = {
        "STGAT+MDN": (np.mean(ade_stgat), np.mean(fde_stgat)),
        "Const-Vel": (np.mean(ade_cv), np.mean(fde_cv)),
        "LSTM base": (np.mean(ade_lstm), np.mean(fde_lstm)),
    }

    best_name = min(models, key=lambda k: models[k][0])

    # -------------------- 
    # Print Results 
    # --------------------
    print()
    print(f"{BOLD}=== Evaluation (Test Set) ==={RESET}")
    print(f"{BOLD}Model         |    ADE    |    FDE   {RESET}")
    print("--------------------------------------")

    for name, (ade, fde) in models.items():
        color = GREEN if name == best_name else CYAN
        print(f"{color}{name:12s} | {ade:8.4f} | {fde:8.4f}{RESET}")

    print()
    print(f"{BOLD}Best Model (by ADE): {RESET}{GREEN}{best_name}{RESET}\n")

    # -------------------- 
    # Interpretation 
    # --------------------
    print(f"{BOLD}How to interpret ADE/FDE:{RESET}")
    print(f"  • {CYAN}ADE{RESET}: Average path accuracy across all future steps.")
    print(f"  • {CYAN}FDE{RESET}: Accuracy of the final predicted position.")
    print("    Lower = better.\n")

    print(f"{BOLD}Model roles:{RESET}")
    print(f"  • {CYAN}STGAT+MDN{RESET}: Main multi-modal trajectory predictor.")
    print(f"  • {CYAN}Const-Vel{RESET}: Baseline using linear extrapolation.")
    print(f"  • {CYAN}LSTM base{RESET}: Baseline without graph attention or map encoding.\n")


if __name__ == "__main__":
    evaluate()
