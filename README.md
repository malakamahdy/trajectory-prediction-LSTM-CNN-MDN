## Pedestrian Trajectory Prediction using Spatio-Temporal Graph Attention Networks (ST-GAT) ◡̈

This project implements a Spatio-Temporal Graph Attention Network (ST-GAT) for multi-modal pedestrian trajectory prediction on the ETH/UCY datasets.

**The model predicts 12 future (x, y) positions given 8 past positions, integrating:**
* Bidirectional LSTM — temporal motion encoding
* Graph Attention Network (GAT) — social interaction modeling
* CNN Map Encoder — lightweight rasterized spatial context
* Mixture Density Network (MDN) — multi-modal Gaussian future predictions

**Two baseline models are included:**
* Constant Velocity baseline (linear extrapolation)
* Simple LSTM baseline (no GAT, no CNN, no MDN)

**Evaluation uses the standard trajectory metrics:**
* ADE — Average Displacement Error
* FDE — Final Displacement Error

### Project Structure

```├── main.py # Train or continue training ST-GAT
├── train.py # ST-GAT training loop
├── train_lstm_baseline.py # Train simple LSTM baseline
├── eval.py # Evaluate ST-GAT, LSTM, CV baselines
├── model.py # ST-GAT + CNN encoder + MDN decoder
├── baselines.py # Baseline models (Const-Vel, LSTM)
├── dataset.py # Dataset loader + preprocessing
├── utils.py # MDN loss, ADE/FDE metrics
│
├── pretrained_model/ # Saved ST-GAT weights
│ └── stgat_mdn.pth
│
├── pretrained_baseline_model/ # Saved LSTM baseline weights
│ └── lstm_baseline.pth
│
├── data/ # ETH/UCY dataset CSVs (not included)
└── README.md
```
*Note: The pretrained models are available. You may train your own from scratch, selected from ```main.py```.*

### Installation
Install required libraries:

```pip install torch numpy pandas```

### How to Run
**1. Train or Continue Training the ST-GAT Model**

```python main.py```

You will be prompted:

```
(c) Continue training from pretrained_model/stgat_mdn.pth
(n) Train new ST-GAT model from scratch
```
* Choose n to start over
* Choose c to continue training from your saved checkpoint (there is a pretrained model available, you can use that as a starting point or delete)

**2. Train the LSTM Baseline**
```
python train_lstm_baseline.py
```
This trains the baseline simple LSTM model (used for comparison).

**3. Evaluate All Models (ST-GAT, LSTM, CV)**
```
python eval.py
```

Example output:

```
Model         |    ADE    |    FDE
--------------------------------------
STGAT+MDN     |   4.46    |   5.14
Const-Vel     |   0.50    |   1.10
LSTM base     |   0.58    |   1.14
```

### Evaluation Metrics Explained
**ADE — Average Displacement Error**

Average L2 distance between predicted and true coordinates across all predicted timesteps.

**FDE — Final Displacement Error**

L2 distance between only the final predicted point and the final true point.

### Model Summary
**ST-GAT (Main Proposed Model)**
* Temporal modeling via Bidirectional LSTM
* Spatial social interaction modeling via GAT
* Map context via CNN encoder
* Multi-modal MDN output (Gaussian mixture)

**Baselines**

* **Constant Velocity:** simple linear extrapolation baseline

* **LSTM baseline:** no GAT, no CNN, deterministic predictions

