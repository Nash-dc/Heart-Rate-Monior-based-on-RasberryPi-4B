# -*- coding: utf-8 -*-
"""
Pulse‑Fi Model Training Script
==============================

This script implements a complete training pipeline for reproducing the
heart‑rate estimation model described in the Pulse‑Fi paper.  It assumes
you have already extracted Wi‑Fi CSI data from Nexmon PCAP files and
matched them with corresponding smartwatch heart‑rate recordings.  The
expected directory structure is the following:

```
project_root/
├── CleanData/
│   ├── 000/
│   │   ├── 1_2022_03_29_-_11_06_33_bw_80_ch_36_csi234.npy
│   │   ├── 1_2022_03_29_-_11_06_33_bw_80_ch_36_meta.json
│   │   └── …
│   └── ...
└── Data_DS1_smartwatch/
    └── Data/
        ├── 000/
        │   ├── 1_2022_03_29_-_11_06_33_HeartRateData.json
        │   └── …
        └── ...
```

Each `*_csi234.npy` file contains a two‑dimensional array of shape
`(num_frames, 234)`, representing the complex CSI values per frame for
234 valid subcarriers.  The accompanying `*_meta.json` file should
provide metadata including the participant ID, activity code, timestamp
and the name of the corresponding heart‑rate JSON file.  Heart‑rate
JSON files must define two keys:

```
{
  "heart_rate": [88.0, 98.0, ...],
  "start_time": ["2022-03-29 11:08:34.995000", "2022-03-29 11:08:44.995000", ...]
}
```

The script will automatically align heart‑rate readings to the CSI
sequence using linear interpolation, apply the same pre‑processing
described in the Pulse‑Fi paper (DC removal, Butterworth bandpass
filter, Savitzky–Golay smoothing, sliding‑window segmentation and
z‑score normalisation), and train a lightweight LSTM regression
network.  The resulting model is saved as `pulsefi_model.pt`.

Requirements
------------
* Python ≥ 3.8
* numpy
* scipy
* torch

Usage
-----
Run the script from the command line, specifying the top‑level
directories for your clean CSI data and smartwatch data:

```
python train_pulsefi.py --csi-dir /path/to/CleanData \
                       --watch-dir /path/to/Data_DS1_smartwatch/Data \
                       --output pulsefi_model.pt
```

You can adjust window length, hop length and training hyperparameters
via command‑line options.  Use `python train_pulsefi.py --help` for a
complete list of arguments.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the Pulse‑Fi LSTM model on pre‑processed CSI data."
    )
    parser.add_argument(
        "--csi-dir",
        type=str,
        required=True,
        help="Path to the root directory containing CleanData/<subject_id> folders with CSI and meta files.",
    )
    parser.add_argument(
        "--watch-dir",
        type=str,
        required=True,
        help="Path to the Data_DS1_smartwatch/Data directory containing heart‑rate JSON files organised by subject.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pulsefi_model.pt",
        help="File path to save the trained model (default: pulsefi_model.pt)",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=20.0,
        help="Sliding window length in seconds (default: 20 seconds)",
    )
    parser.add_argument(
        "--hop-sec",
        type=float,
        default=1.0,
        help="Sliding window hop length in seconds (default: 1 second)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini‑batch size for training (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of epochs to train (default: 50)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Number of hidden units in the LSTM (default: 128)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Number of LSTM layers (default: 1)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10 epochs)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (default: cuda if available else cpu)",
    )
    return parser.parse_args()


def read_heart_rate(json_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read heart‑rate data and timestamps from a JSON file.

    Parameters
    ----------
    json_path : Path
        Path to the heart‑rate JSON file.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple (heart_rates, times) where `heart_rates` is a 1‑D array of
        heart‑rate values (floats) and `times` is an array of relative
        times (seconds) measured from the first timestamp.
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    hr = np.asarray(data["heart_rate"], dtype=float)
    # Convert timestamp strings to datetime objects
    times = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f") for ts in data["start_time"]]
    base = times[0]
    rel_times = np.array([(t - base).total_seconds() for t in times], dtype=float)
    return hr, rel_times


def bandpass_filter(data: np.ndarray, fs: float, low: float = 0.8, high: float = 2.17, order: int = 3) -> np.ndarray:
    """Apply a Butterworth bandpass filter to each subcarrier.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (frames, subcarriers) containing CSI values.
    fs : float
        Sampling frequency in Hz.
    low : float
        Lower cutoff frequency in Hz (default 0.8 Hz).
    high : float
        Upper cutoff frequency in Hz (default 2.17 Hz).
    order : int
        Filter order (default 3).

    Returns
    -------
    np.ndarray
        Filtered data of the same shape.
    """
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    # Ensure cutoff frequencies are within (0, 1)
    if high_norm >= 1.0:
        high_norm = 0.99
    b, a = butter(order, [low_norm, high_norm], btype="band")
    return filtfilt(b, a, data, axis=0)


def smooth_signal(data: np.ndarray, window_length: int = 15, polyorder: int = 3) -> np.ndarray:
    """Apply Savitzky–Golay smoothing along the time axis.

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (frames, subcarriers).
    window_length : int
        Length of the filter window (default 15). Must be an odd integer.
    polyorder : int
        Polynomial order to use in the filtering (default 3).

    Returns
    -------
    np.ndarray
        Smoothed array of the same shape.
    """
    # Window length must not exceed the number of frames and must be odd
    wl = window_length
    if wl >= data.shape[0]:
        wl = data.shape[0] - 1 if data.shape[0] % 2 == 0 else data.shape[0]
    if wl % 2 == 0:
        wl -= 1
    return savgol_filter(data, window_length=wl, polyorder=polyorder, axis=0)


def create_windows(
    signal: np.ndarray,
    labels: np.ndarray,
    fs: float,
    window_sec: float,
    hop_sec: float,
) -> Tuple[List[np.ndarray], List[float]]:
    """Segment the signal and labels into overlapping sliding windows.

    Each window is z‑score normalised along the time axis and labelled
    with the mean heart rate within that window.

    Parameters
    ----------
    signal : np.ndarray
        Array of shape (frames, subcarriers).
    labels : np.ndarray
        Array of shape (frames,) containing heart‑rate values per frame.
    fs : float
        Sampling frequency (Hz).
    window_sec : float
        Length of each window in seconds.
    hop_sec : float
        Hop length between windows in seconds.

    Returns
    -------
    Tuple[List[np.ndarray], List[float]]
        Lists of windowed signals and their corresponding labels.
    """
    window_len = int(round(window_sec * fs))
    hop_len = int(round(hop_sec * fs))
    xs: List[np.ndarray] = []
    ys: List[float] = []
    num_frames = signal.shape[0]
    if window_len < 1 or hop_len < 1:
        return xs, ys
    for start in range(0, num_frames - window_len + 1, hop_len):
        end = start + window_len
        seg = signal[start:end]
        lbl_seg = labels[start:end]
        # Z‑score normalisation per subcarrier
        seg_norm = (seg - seg.mean(axis=0)) / (seg.std(axis=0) + 1e-8)
        xs.append(seg_norm.astype(np.float32))
        ys.append(float(lbl_seg.mean()))
    return xs, ys


class PulseDataset(Dataset):
    """Custom Dataset for windowed CSI data and heart‑rate labels."""

    def __init__(self, windows: List[np.ndarray], labels: List[float]):
        assert len(windows) == len(labels)
        self.data = torch.tensor(np.stack(windows), dtype=torch.float32)
        self.targets = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


class LSTMRegressor(nn.Module):
    """Lightweight LSTM network for heart‑rate regression."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        output, _ = self.lstm(x)
        # Use the last time step's hidden state
        last_output = output[:, -1, :]
        return self.fc(last_output).squeeze(1)


def prepare_dataset(
    csi_dir: Path, watch_dir: Path, window_sec: float, hop_sec: float
) -> Tuple[List[np.ndarray], List[float]]:
    """Load and process all sessions into windows and labels.

    Parameters
    ----------
    csi_dir : Path
        Root directory of processed CSI data organised by subject.
    watch_dir : Path
        Root directory of smartwatch data organised by subject.
    window_sec : float
        Window length in seconds.
    hop_sec : float
        Hop length in seconds.

    Returns
    -------
    Tuple[List[np.ndarray], List[float]]
        Aggregated lists of windowed signals and labels from all sessions.
    """
    all_windows: List[np.ndarray] = []
    all_labels: List[float] = []
    for subj_path in sorted(csi_dir.iterdir()):
        if not subj_path.is_dir():
            continue
        subject_id = subj_path.name
        # Each subject may have multiple sessions
        for meta_path in sorted(subj_path.glob("*_meta.json")):
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            hr_file = meta.get("heart_rate_file")
            if not hr_file:
                # Skip sessions without heart‑rate reference
                continue
            # Paths to CSI and heart‑rate data
            csi_file = subj_path / meta["output_file"]
            watch_file = watch_dir / subject_id / hr_file
            if not csi_file.exists() or not watch_file.exists():
                continue
            # Load CSI and heart‑rate
            csi = np.load(csi_file)  # shape: (num_frames, 234), complex dtype
            # Use amplitude only
            csi_amp = np.abs(csi).astype(np.float32)
            hr, hr_times = read_heart_rate(watch_file)
            # Determine sampling interval for CSI frames
            num_frames = csi_amp.shape[0]
            if len(hr_times) < 2:
                # Not enough HR data to interpolate
                continue
            duration = hr_times[-1] - hr_times[0]
            if duration <= 0:
                continue
            fs = num_frames / duration  # frames per second
            # Align heart‑rate to each frame via linear interpolation
            frame_times = np.linspace(hr_times[0], hr_times[-1], num_frames)
            hr_per_frame = np.interp(frame_times, hr_times, hr)
            # DC removal
            csi_dc_removed = csi_amp - csi_amp.mean(axis=0, keepdims=True)
            # Bandpass filtering and smoothing
            csi_filtered = bandpass_filter(csi_dc_removed, fs)
            csi_smoothed = smooth_signal(csi_filtered)
            # Create windows
            windows, labels = create_windows(
                csi_smoothed, hr_per_frame, fs, window_sec, hop_sec
            )
            all_windows.extend(windows)
            all_labels.extend(labels)
    # If no windows were collected, return immediately
    if not all_windows:
        return all_windows, all_labels
    # ------------------------------------------------------------------
    # Some sessions may have been sampled at different rates, resulting
    # in windows with varying frame lengths.  Stacking arrays of
    # different lengths will fail later, so here we resample every
    # window to a common frame length.  We choose the minimum window
    # length across all collected segments to minimise information
    # loss.  Each segment is linearly interpolated along the time axis
    # to match the target length.
    lengths = [w.shape[0] for w in all_windows]
    target_len = min(lengths)
    if len(set(lengths)) > 1:
        print(
            f"Resampling windows to a common length of {target_len} frames (from lengths {sorted(set(lengths))})."
        )
        resampled_windows: List[np.ndarray] = []
        for seg in all_windows:
            old_len, n_sub = seg.shape
            if old_len == target_len:
                resampled_windows.append(seg.astype(np.float32))
                continue
            # Normalise x-axis to [0, 1] for interpolation
            x_old = np.linspace(0.0, 1.0, old_len)
            x_new = np.linspace(0.0, 1.0, target_len)
            resampled = np.empty((target_len, n_sub), dtype=np.float32)
            for j in range(n_sub):
                resampled[:, j] = np.interp(x_new, x_old, seg[:, j])
            resampled_windows.append(resampled)
        all_windows = resampled_windows
    return all_windows, all_labels


def train_model(
    dataset: PulseDataset,
    hidden_dim: int,
    num_layers: int,
    batch_size: int,
    epochs: int,
    patience: int,
    device: str,
    output_path: str,
) -> None:
    """Train the LSTM model using the provided dataset and save it.

    Parameters
    ----------
    dataset : PulseDataset
        Dataset containing training samples and labels.
    hidden_dim : int
        Hidden dimension of the LSTM.
    num_layers : int
        Number of stacked LSTM layers.
    batch_size : int
        Batch size used for DataLoader.
    epochs : int
        Maximum number of training epochs.
    patience : int
        Early stopping patience.
    device : str
        Device identifier, e.g., 'cpu' or 'cuda'.
    output_path : str
        Path where the trained model will be saved.
    """
    # Split dataset into train/val/test (64/16/20)
    total_len = len(dataset)
    train_len = int(0.64 * total_len)
    val_len = int(0.16 * total_len)
    test_len = total_len - train_len - val_len
    train_set, val_set, test_set = random_split(
        dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    # Initialise model
    input_dim = dataset.data.shape[2]
    model = LSTMRegressor(input_dim, hidden_dim, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )
    best_val_loss = float("inf")
    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        print(
            f"Epoch {epoch:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={optimizer.param_groups[0]['lr']:.1e}"
        )
        # Early stopping
        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save best model
            torch.save(model.state_dict(), output_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"No improvement for {patience} epochs, stopping training.")
                break
    # Load best model for final evaluation
    model.load_state_dict(torch.load(output_path, map_location=device))
    model.eval()
    # Evaluate on test set
    test_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_x.size(0)
    test_loss /= len(test_loader.dataset)
    print(f"Test loss (MSE): {test_loss:.6f}")


def main() -> None:
    args = parse_args()
    csi_dir = Path(args.csi_dir)
    watch_dir = Path(args.watch_dir)
    # Load and process data
    print("Loading and preprocessing data...")
    windows, labels = prepare_dataset(csi_dir, watch_dir, args.window_sec, args.hop_sec)
    if not windows:
        raise RuntimeError("No valid windows generated. Please check your data paths and meta files.")
    dataset = PulseDataset(windows, labels)
    print(f"Total windows: {len(dataset)}")
    # Train the model
    print("Training model...")
    train_model(
        dataset,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        output_path=args.output,
    )
    print(f"Training complete. Model saved to {args.output}")


if __name__ == "__main__":
    main()