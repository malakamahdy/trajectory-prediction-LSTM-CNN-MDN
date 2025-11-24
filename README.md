## Pedestrian Trajectory Prediction using LSTM-CNN-MDN ◡̈

This project implements an LSTM-CNN-MDN model for multi-modal pedestrian trajectory prediction on the ETH/UCY datasets.

**The model predicts 12 future (x, y) positions given 8 past positions, integrating:**
* Bidirectional LSTM — temporal motion encoding
* CNN Map Encoder — lightweight rasterized spatial context
* Mixture Density Network (MDN) — multi-modal Gaussian future predictions

**Two baseline models are included:**
* Constant Velocity baseline (linear extrapolation)
* Simple LSTM baseline (no CNN, no MDN)

**Evaluation uses the standard trajectory metrics:**
* ADE — Average Displacement Error
* FDE — Final Displacement Error

### Project Structure
```
├── main.py                      # Train or continue training LSTM-CNN-MDN
├── train.py                     # LSTM-CNN-MDN training loop
├── train_lstm_baseline.py       # Train simple LSTM baseline
├── eval.py                      # Evaluate LSTM-CNN-MDN, LSTM, CV baselines
├── model.py                     # LSTM-CNN-MDN + CNN encoder + MDN decoder
├── baselines.py                 # Baseline models (Const-Vel, LSTM)
├── dataset.py                   # Dataset loader + preprocessing
├── utils.py                     # MDN loss, ADE/FDE metrics
│
├── pretrained_model/            # Saved model weights 
│   └── lstmcnnmdn.pth           
│
├── pretrained_baseline_model/   # Saved LSTM baseline weights
│   └── lstm_baseline.pth        
│
├── data/                        # ETH/UCY dataset CSVs 
└── README.md
```

*Note: Pretrained models are available. You may train your own from scratch via `main.py`.*

### Installation
Install required libraries:
```bash
pip install torch numpy pandas
```

### How to Run

**1. Train or Continue Training the LSTM-CNN-MDN Model**
```bash
python main.py
```

You will be prompted:
```
(c) Continue training from existing: pretrained_model/lstmcnnmdn.pth
(n) Train new LSTM-CNN-MDN model from scratch
```
* Choose `n` to start over (will override existing model)
* Choose `c` to continue training from saved checkpoint

**2. Train the LSTM Baseline**
```bash
python train_lstm_baseline.py
```

This trains the baseline simple LSTM model (used for comparison).

**3. Evaluate All Models (LSTM-CNN-MDN, LSTM, CV)**
```bash
python eval.py
```

Example output:
```
Model         |    ADE    |    FDE
--------------------------------------
LSTM-CNN-MDN  |   0.48    |   1.00
Const-Vel     |   0.50    |   1.10
LSTM base     |   0.59    |   1.11
```

### Evaluation Metrics Explained

**ADE — Average Displacement Error**

Average L2 distance between predicted and true coordinates across all predicted timesteps.

**FDE — Final Displacement Error**

L2 distance between only the final predicted point and the final true point.

### Model Summary

**LSTM-CNN-MDN (Main Proposed Model)**
* Temporal modeling via Bidirectional LSTM
* Map context via CNN encoder
* Multi-modal MDN output (Gaussian mixture)

**Baselines**
* **Constant Velocity:** simple linear extrapolation baseline
* **LSTM baseline:** no CNN, no MDN, deterministic predictions