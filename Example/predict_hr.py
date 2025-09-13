#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pulse-Fi Single File Prediction Script
======================================

This script loads a trained Pulse-Fi LSTM model and uses it to predict heart rate from a single CSI data file.
It reads a CSI amplitude data file (.npy format, shape (frames, 234)), applies the same preprocessing as in Pulse-Fi:
DC removal, Butterworth bandpass filter (0.8–2.17 Hz), Savitzky-Golay smoothing, and segments the data into 20-second windows (with 1s hop).
Each window is z-score normalized and fed into the model to estimate heart rate (in beats per minute).
The script outputs the predicted heart rate for each window and the average heart rate over the entire input.

Usage:
    python predict_single_file.py --input csi_234.npy --model best_model.pt

Requirements:
    numpy, scipy, torch

Model architecture:
    LSTM (input_dim = 234, hidden_dim = 128, num_layers = 1), followed by a fully-connected layer to predict a single value.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt, savgol_filter

# Define the LSTM regression model (same structure as used in training)
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                             num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        # take output of last time step
        last_out = out[:, -1, :]
        # final linear layer to single output
        return self.fc(last_out).squeeze(1)

# Butterworth bandpass filter (applied to each subcarrier time-series)
def bandpass_filter(data: np.ndarray, fs: float, low: float = 0.8, high: float = 2.17, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    if high_norm >= 1.0:
        high_norm = 0.99  # ensure upper cutoff is below Nyquist
    b, a = butter(order, [low_norm, high_norm], btype="band")
    # Apply zero-phase bandpass filter along time axis (axis=0)
    return filtfilt(b, a, data, axis=0)

# Savitzky-Golay smoothing filter (applied to each subcarrier time-series)
def smooth_signal(data: np.ndarray, window_length: int = 15, polyorder: int = 3) -> np.ndarray:
    wl = window_length
    # window length must be <= number of frames and odd
    if wl >= data.shape[0]:
        wl = data.shape[0] - 1 if data.shape[0] % 2 == 0 else data.shape[0]
    if wl % 2 == 0:
        wl -= 1
    if wl < 1 or wl <= polyorder:
        # If data is too short, just return original data (no smoothing possible)
        return data
    return savgol_filter(data, window_length=wl, polyorder=polyorder, axis=0)

def main():
    parser = argparse.ArgumentParser(description="Predict heart rate from a single CSI file using Pulse-Fi model")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input .npy file containing CSI amplitude data")
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to trained model (.pt file)")
    parser.add_argument("--device", "-d", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run model on (cpu or cuda). Default: use cuda if available, else cpu.")
    args = parser.parse_args()

    device = torch.device(args.device)
    # Step 1: Load CSI amplitude data from .npy file
    csi_data = np.load(args.input)
    if csi_data.size == 0:
        print("Error: Input CSI data is empty.")
        return
    # If complex values, convert to amplitude
    if np.iscomplexobj(csi_data):
        csi_amp = np.abs(csi_data).astype(np.float32)
    else:
        csi_amp = csi_data.astype(np.float32)

    # Step 2: Define sampling frequency (fs) from sampling interval 0.136s (~7.35 Hz)
    fs = 1.0 / 0.136  # ~7.35 Hz
    # Step 3: Preprocess the CSI data
    # 3.1 Remove DC component (zero-mean per subcarrier)
    csi_dc = csi_amp - np.mean(csi_amp, axis=0, keepdims=True)
    # 3.2 Bandpass filter (0.8-2.17 Hz) to isolate heart rate frequencies
    csi_bp = bandpass_filter(csi_dc, fs)
    # 3.3 Savitzky-Golay smoothing to reduce high-frequency noise
    csi_smoothed = smooth_signal(csi_bp)

    # Step 4: Sliding window segmentation (20-second windows with 1-second hop)
    window_sec = 20.0
    hop_sec = 1.0
    window_len = int(round(window_sec * fs))
    hop_len = int(round(hop_sec * fs))
    frames = csi_smoothed.shape[0]
    windows = []
    for start in range(0, frames - window_len + 1, hop_len):
        end = start + window_len
        segment = csi_smoothed[start:end]
        # Step 5: Per-window z-score normalization
        seg_norm = (segment - segment.mean(axis=0)) / (segment.std(axis=0) + 1e-8)
        windows.append(seg_norm.astype(np.float32))
    if not windows:
        # If no window was collected (data shorter than one window length), use the whole sequence as one window
        segment = csi_smoothed
        seg_norm = (segment - segment.mean(axis=0)) / (segment.std(axis=0) + 1e-8)
        windows.append(seg_norm.astype(np.float32))
        print(f"Warning: Input data shorter than {window_sec} seconds. Using entire sequence ({frames} frames, ~{frames/fs:.1f}s) as one window.")

    # Convert list of windows to tensor
    # (Each window is shape [window_len, num_subcarriers])
    input_dim = windows[0].shape[1]  # number of subcarriers (features)
    # Step 6: Load the trained model
    model = LSTMRegressor(input_dim=input_dim, hidden_dim=128, num_layers=1).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Step 7: Predict heart rate for each window
    preds = []
    with torch.no_grad():
        for seg in windows:
            X = torch.from_numpy(seg).unsqueeze(0).to(device)  # shape (1, window_len, input_dim)
            y_pred = model(X)
            # y_pred is a tensor of shape (1,) (single value)
            preds.append(float(y_pred.cpu().item()))
    preds = np.array(preds)
    avg_hr = float(np.mean(preds))

    # Output the results
    # Print predicted heart rates for each window and the average heart rate
    # Format predictions to one decimal place for clarity
    pred_str = ", ".join(f"{p:.1f}" for p in preds)
    print(f"Predicted heart rate for each window (bpm): [{pred_str}]")
    print(f"Average heart rate over the entire segment (bpm): {avg_hr:.1f}")

if __name__ == "__main__":
    main()
