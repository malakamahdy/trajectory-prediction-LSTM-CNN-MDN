from train import train_model
import os

# ANSI color helpers
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"

# Model folders
MODEL_DIR = "pretrained_model"
BASELINE_DIR = "pretrained_baseline_model"

# File paths
MODEL_PATH = os.path.join(MODEL_DIR, "stgat_mdn.pth")
BASELINE_PATH = os.path.join(BASELINE_DIR, "lstm_baseline.pth")

def banner():
    print(f"{MAGENTA}{BOLD}")
    print("===============================================")
    print("  Pedestrian Trajectory Prediction (ST-GAT)")
    print("===============================================")
    print(f"{RESET}")
    print(f"{CYAN}Model:{RESET} Bidirectional LSTM + Graph Attention + CNN Map Encoder + MDN")
    print(f"{CYAN}Baselines:{RESET} Constant Velocity, Simple LSTM")
    print()
    print(f"{BOLD}How to use this project:{RESET}")
    print(f"  1) Train or continue training the main ST-GAT model:")
    print(f"     {CYAN}python main.py{RESET}")
    print(f"     → choose {BOLD}n{RESET} to start from scratch, or {BOLD}c{RESET} to continue training.")
    print()
    print(f"  2) Train the simple LSTM baseline (no graph attention):")
    print(f"     {CYAN}python train_lstm_baseline.py{RESET}")
    print()
    print(f"  3) Evaluate all models (ST-GAT, Const-Vel, LSTM) with ADE/FDE:")
    print(f"     {CYAN}python eval.py{RESET}")
    print()
    print(f"  ADE = Average Displacement Error (overall trajectory error)")
    print(f"  FDE = Final Displacement Error (error at final point)")
    print()

if __name__ == "__main__":

    # Ensure folders exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(BASELINE_DIR, exist_ok=True)

    banner()

    print(f"{BOLD}Select an option:{RESET}")
    print(f"  {CYAN}(c){RESET} Continue training from existing: {BOLD}{MODEL_PATH}{RESET}")
    print(f"  {CYAN}(n){RESET} Train {BOLD}new ST-GAT model{RESET} from scratch")
    choice = input("Enter c or n: ").strip().lower()

    if choice == "c":
        if not os.path.exists(MODEL_PATH):
            print(f"{YELLOW}⚠ No saved model found at {MODEL_PATH}. Starting new training instead.{RESET}")
            train_model(load_existing=False, model_path=MODEL_PATH)
        else:
            print(f"{GREEN}Loading existing model: {MODEL_PATH}{RESET}")
            train_model(load_existing=True, model_path=MODEL_PATH)
    else:
        print(f"{YELLOW}Starting a fresh training run...{RESET}")
        train_model(load_existing=False, model_path=MODEL_PATH)
