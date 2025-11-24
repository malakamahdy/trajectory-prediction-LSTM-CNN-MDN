import torch
from torch.utils.data import Dataset
import pandas as pd
import glob
import os
import numpy as np

class TrajectoryDataset(Dataset):
    def __init__(self, data_folder="data", history=8, future=12, verbose=True):
        """
        history: number of past timesteps
        future: number of future timesteps

        Assumes CSV format with NO HEADER and 4 columns:
            col0 = frame (time step)
            col1 = agent_id
            col2 = x coordinate
            col3 = y coordinate
        """
        self.history = history
        self.future = future
        self.samples = []

        csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
        if verbose:
            print(f"Found {len(csv_files)} CSV files in '{data_folder}':")
            for f in csv_files:
                print("  -", os.path.basename(f))

        for file_path in csv_files:
            if verbose:
                print(f"\n=== Loading {os.path.basename(file_path)} ===")

            df = pd.read_csv(file_path, header=None)

            # Expect at least 4 columns: frame, agent_id, x, y
            if df.shape[1] < 4:
                print(f"Skipping {os.path.basename(file_path)}: fewer than 4 columns.")
                continue

            df = df.iloc[:, :4]
            df.columns = ["frame", "agent_id", "x", "y"]

            if verbose:
                print("First few rows:")
                print(df.head())

            # Group by agent (pedestrian/vehicle id)
            for agent_id, agent_df in df.groupby("agent_id"):
                agent_df = agent_df.sort_values("frame")
                coords = agent_df[["x", "y"]].to_numpy()

                if len(coords) >= history + future:
                    for start in range(len(coords) - (history + future) + 1):
                        past = coords[start:start + history]            # [H,2]
                        fut = coords[start + history:start + history + future]  # [F,2]
                        self.samples.append((past, fut))

        if verbose:
            print(f"\n Total training samples constructed: {len(self.samples)}")
            if len(self.samples) == 0:
                print(" No samples were created. Check that your CSVs really contain enough frames per agent.")

    @staticmethod
    def build_traj_map(past, H=32, W=32):
        """
        Build a simple raster "map" from past trajectory:
        - normalize x,y to [0, W-1] x [0, H-1]
        - mark visited cells as 1.0
        - return 3xHxW (fake RGB)
        """
        past = np.asarray(past)  # [H,2]
        x = past[:, 0]
        y = past[:, 1]

        # Avoid divide-by-zero: if trajectory is flat, just keep zeros
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        if x_range < 1e-6:
            x_range = 1.0
        if y_range < 1e-6:
            y_range = 1.0

        x_norm = (x - x.min()) / x_range
        y_norm = (y - y.min()) / y_range

        xi = np.clip((x_norm * (W - 1)).astype(int), 0, W - 1)
        yi = np.clip((y_norm * (H - 1)).astype(int), 0, H - 1)

        grid = np.zeros((H, W), dtype=np.float32)
        grid[yi, xi] = 1.0

        # 3 channels (fake RGB)
        grid3 = np.stack([grid, grid, grid], axis=0)  # [3,H,W]
        return grid3

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        past, fut = self.samples[idx]

        map_arr = self.build_traj_map(past)          # [3,32,32]

        past = torch.tensor(past, dtype=torch.float32)   # [H,2]
        fut = torch.tensor(fut, dtype=torch.float32)     # [F,2]
        map_tensor = torch.tensor(map_arr, dtype=torch.float32)

        return past, fut, map_tensor
