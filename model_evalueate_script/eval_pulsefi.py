#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pulse-Fi Model Evaluation Script
================================

This script loads a trained Pulse-Fi LSTM model and evaluates it on the
test split of the dataset. It prints MSE, RMSE, MAE and plots predicted
vs. ground-truth heart rate curves.

Usage
-----
python eval_pulsefi.py --csi-dir /path/to/CleanData \
                       --watch-dir /path/to/Data_DS1_smartwatch/Data \
                       --model pulsefi_model.pt \
                       --hidden-dim 128 \
                       --num-layers 1
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, savgol_filter
from datetime import datetime


# -----------------------------
# Model definition
# -----------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)


# -----------------------------
# Dataset utilities
# -----------------------------
class PulseDataset(Dataset):
    def __init__(self, windows, labels):
        self.data = torch.tensor(np.stack(windows), dtype=torch.float32)
        self.targets = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def read_heart_rate(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    hr = np.asarray(data["heart_rate"], dtype=float)
    times = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f") for ts in data["start_time"]]
    base = times[0]
    rel_times = np.array([(t - base).total_seconds() for t in times], dtype=float)
    return hr, rel_times


def bandpass_filter(data, fs, low=0.8, high=2.17, order=3):
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = min(high / nyq, 0.99)
    b, a = butter(order, [low_norm, high_norm], btype="band")
    return filtfilt(b, a, data, axis=0)


def smooth_signal(data, window_length=15, polyorder=3):
    wl = window_length
    if wl >= data.shape[0]:
        wl = data.shape[0] - 1 if data.shape[0] % 2 == 0 else data.shape[0]
    if wl % 2 == 0:
        wl -= 1
    return savgol_filter(data, window_length=wl, polyorder=polyorder, axis=0)


def create_windows(signal, labels, fs, window_sec, hop_sec):
    window_len = int(round(window_sec * fs))
    hop_len = int(round(hop_sec * fs))
    xs, ys = [], []
    num_frames = signal.shape[0]
    for start in range(0, num_frames - window_len + 1, hop_len):
        end = start + window_len
        seg = signal[start:end]
        lbl_seg = labels[start:end]
        seg_norm = (seg - seg.mean(axis=0)) / (seg.std(axis=0) + 1e-8)
        xs.append(seg_norm.astype(np.float32))
        ys.append(float(lbl_seg.mean()))
    return xs, ys


def prepare_dataset(csi_dir, watch_dir, window_sec, hop_sec):
    all_windows, all_labels = [], []
    for subj_path in sorted(Path(csi_dir).iterdir()):
        if not subj_path.is_dir():
            continue
        subject_id = subj_path.name
        for meta_path in sorted(subj_path.glob("*_meta.json")):
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            hr_file = meta.get("heart_rate_file")
            if not hr_file:
                continue
            csi_file = subj_path / meta["output_file"]
            watch_file = Path(watch_dir) / subject_id / hr_file
            if not csi_file.exists() or not watch_file.exists():
                continue
            csi = np.load(csi_file)
            csi_amp = np.abs(csi).astype(np.float32)
            hr, hr_times = read_heart_rate(watch_file)
            num_frames = csi_amp.shape[0]
            if len(hr_times) < 2:
                continue
            duration = hr_times[-1] - hr_times[0]
            if duration <= 0:
                continue
            fs = num_frames / duration
            frame_times = np.linspace(hr_times[0], hr_times[-1], num_frames)
            hr_per_frame = np.interp(frame_times, hr_times, hr)
            csi_dc_removed = csi_amp - csi_amp.mean(axis=0, keepdims=True)
            csi_filtered = bandpass_filter(csi_dc_removed, fs)
            csi_smoothed = smooth_signal(csi_filtered)
            windows, labels = create_windows(csi_smoothed, hr_per_frame, fs, window_sec, hop_sec)
            all_windows.extend(windows)
            all_labels.extend(labels)
    # unify length
    if not all_windows:
        return [], []
    target_len = min(w.shape[0] for w in all_windows)
    resampled = []
    for seg in all_windows:
        old_len, n_sub = seg.shape
        if old_len == target_len:
            resampled.append(seg)
            continue
        x_old = np.linspace(0, 1, old_len)
        x_new = np.linspace(0, 1, target_len)
        seg_new = np.stack([np.interp(x_new, x_old, seg[:, j]) for j in range(n_sub)], axis=1)
        resampled.append(seg_new)
    return resampled, all_labels


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csi-dir", required=True)
    parser.add_argument("--watch-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--window-sec", type=float, default=20.0)
    parser.add_argument("--hop-sec", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=128, help="Must match training")
    parser.add_argument("--num-layers", type=int, default=1, help="Must match training")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("[INFO] Loading dataset...")
    windows, labels = prepare_dataset(args.csi_dir, args.watch_dir, args.window_sec, args.hop_sec)
    if not windows:
        print("No data found!")
        return

    dataset = PulseDataset(windows, labels)
    n = len(dataset)
    n_train = int(0.64 * n)
    n_val = int(0.16 * n)
    n_test = n - n_train - n_val
    _, _, test_dataset = random_split(dataset, [n_train, n_val, n_test])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # load model
    input_dim = windows[0].shape[1]
    model = LSTMRegressor(input_dim, args.hidden_dim, args.num_layers).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y_pred = model(X).cpu().numpy().flatten()
            preds.extend(y_pred)
            trues.extend(y.numpy().flatten())

    preds = np.array(preds)
    trues = np.array(trues)
    mse = np.mean((preds - trues) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - trues))
    print(f"[RESULT] Test MSE={mse:.3f}, RMSE={rmse:.3f} bpm, MAE={mae:.3f} bpm")

    plt.figure(figsize=(10, 4))
    plt.plot(trues[:500], label="True HR", linewidth=2)
    plt.plot(preds[:500], label="Predicted HR", linewidth=2)
    plt.legend()
    plt.title("Pulse-Fi Prediction vs. Ground Truth")
    plt.xlabel("Sample index")
    plt.ylabel("Heart Rate (bpm)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
