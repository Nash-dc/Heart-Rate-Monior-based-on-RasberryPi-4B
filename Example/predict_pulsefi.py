# -*- coding: utf-8 -*-

import argparse
import numpy as np
import torch
from scipy.signal import butter, filtfilt, savgol_filter, resample

# === Signal preprocessing ===
def bandpass_filter(data, fs, low=0.8, high=2.17, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, data, axis=0)

def smooth_signal(data, wl=15, poly=3):
    return savgol_filter(data, window_length=wl, polyorder=poly, axis=0)

def preprocess_csi(csi, fs):
    # 1. Amplitude
    csi_amp = np.abs(csi).astype(np.float32)
    # 2. Remove DC
    csi_dc = csi_amp - csi_amp.mean(axis=0, keepdims=True)
    # 3. Bandpass filter (0.8–2.17 Hz -> 48–130 bpm)
    csi_bp = bandpass_filter(csi_dc, fs)
    # 4. Savitzky-Golay smoothing
    csi_sm = smooth_signal(csi_bp)
    return csi_sm

# === Resampling ===
def resample_csi(csi, fs_orig, fs_target=7.35):
    new_len = int(len(csi) * fs_target / fs_orig)
    return resample(csi, new_len, axis=0), fs_target

# === LSTM model ===
class LSTMRegressor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=input_dim,
                                  hidden_size=hidden_dim,
                                  num_layers=num_layers,
                                  batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)

# === Main ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input .npy file (frames, 234)")
    parser.add_argument("--model", type=str, required=True, help="Trained model weights (.pt)")
    parser.add_argument("--duration", type=float, required=True, help="Recording duration in seconds")
    args = parser.parse_args()

    # 1. Load CSI
    csi = np.load(args.input)  # shape = (frames, 234)

    # 2. Estimate original sampling rate
    fs_orig = len(csi) / args.duration
    print(f"Original sampling rate: {fs_orig:.2f} Hz")

    # 3. Resample to 7.35 Hz (136 ms)
    csi, fs = resample_csi(csi, fs_orig, fs_target=7.35)
    print(f"Resampled to: {fs:.2f} Hz, new length: {len(csi)}")

    # 4. Preprocess
    proc = preprocess_csi(csi, fs)

    # 5. Sliding window (20s, hop=1s)
    win_sec, hop_sec = 20, 1
    win_len, hop_len = int(win_sec * fs), int(hop_sec * fs)
    windows = []
    for start in range(0, len(proc) - win_len + 1, hop_len):
        seg = proc[start:start+win_len]
        seg_norm = (seg - seg.mean(axis=0)) / (seg.std(axis=0) + 1e-8)
        windows.append(seg_norm.astype(np.float32))
    windows = np.stack(windows)   # (num_windows, win_len, 234)

    # 6. Convert to tensor
    x = torch.tensor(windows, dtype=torch.float32)

    # 7. Load model
    model = LSTMRegressor(input_dim=proc.shape[1])
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    # 8. Predict
    with torch.no_grad():
        preds = model(x).numpy()

    print("Predicted heart rates for each window (bpm):")
    print(preds)
    print("Average heart rate (bpm):", preds.mean())

if __name__ == "__main__":
    main()
