#!/usr/bin/env python3
import os
import json
import numpy as np
from pathlib import Path
from nexcsi import decoder

# Input dirs
DATA_RASP = Path("/home/dacheng/Documents/CaCom/Data_DS1_raspberry/Data")
DATA_WATCH = Path("/home/dacheng/Documents/CaCom/Data_DS1_smartwatch/Data")
OUT_ROOT = Path("/home/dacheng/Documents/CaCom/CleanData")

# Activity mapping
ACTIVITY_MAP = {
    "0": "Empty (no participant in room)",
    "1": "S_F_R (Sitting, facing Raspberry Pi, regular breathing)",
    "2": "S_F_I (Sitting, facing Raspberry Pi, irregular breathing)",
    "3": "SL_F_R (Sit and stand, facing Raspberry Pi, regular breathing)",
    "4": "S_C_R (Sitting, back to Raspberry Pi, regular breathing)",
    "5": "S_C_I (Sitting, back to Raspberry Pi, irregular breathing)",
    "6": "P_F_R (Standing, facing Raspberry Pi, regular breathing)",
    "7": "P_F_I (Standing, facing Raspberry Pi, irregular breathing)",
    "8": "P_C_R (Standing, back to Raspberry Pi, regular breathing)",
    "9": "P_C_I (Standing, back to Raspberry Pi, irregular breathing)",
    "10": "D_F_R (Lying, face up, regular breathing)",
    "11": "D_F_I (Lying, face up, irregular breathing)",
    "12": "D_C_R (Lying, face down, regular breathing)",
    "13": "D_C_I (Lying, face down, irregular breathing)",
    "14": "DL_F_R (Lie down and stand, facing Raspberry Pi, regular breathing)",
    "15": "A_F_R (Walking, facing Raspberry Pi, regular breathing)",
    "16": "C_F_R (Running, facing Raspberry Pi, regular breathing)",
    "17": "V_F_R (Sweeping, facing Raspberry Pi, regular breathing)"
}

print(f"[INFO] Raspberry data dir: {DATA_RASP}")
print(f"[INFO] Smartwatch data dir: {DATA_WATCH}")
print(f"[INFO] Output root: {OUT_ROOT}")

for subject_dir in sorted(DATA_RASP.glob("*")):
    if not subject_dir.is_dir():
        continue

    subj_id = subject_dir.name
    out_dir = OUT_ROOT / subj_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Processing subject {subj_id} ...")

    pcap_files = sorted(subject_dir.glob("*.pcap"))
    print(f"[INFO] Found {len(pcap_files)} pcap files in {subj_id}")

    for pcap_file in pcap_files:
        try:
            print(f"[DEBUG] Parsing {pcap_file.name}")

            # Parse PCAP → CSI
            samples = decoder("raspberrypi").read_pcap(str(pcap_file))
            raw_csi = samples['csi']
            csi_full = decoder("raspberrypi").unpack(raw_csi)

            # Remove Null + Pilot subcarriers
            invalid_idx = sorted(set(csi_full.dtype.metadata['nulls']) |
                                 set(csi_full.dtype.metadata['pilots']))
            csi_clean = np.delete(csi_full, invalid_idx, axis=1)

            # Save CSI array
            out_file = out_dir / (pcap_file.stem + "_csi234.npy")
            np.save(out_file, csi_clean)

            # Match Smartwatch JSON by timestamp
            watch_dir = DATA_WATCH / subj_id
            watch_file = None
            if watch_dir.exists():
                for f in watch_dir.glob("*.json"):
                    # both filenames start with the same activity+date prefix
                    if pcap_file.stem.split("_bw_")[0] in f.stem:
                        watch_file = f.name
                        break

            # Build metadata
            activity_code = pcap_file.stem.split("_")[0]
            activity_name = ACTIVITY_MAP.get(activity_code, "Unknown")

            meta = {
                "subject": subj_id,
                "pcap_file": pcap_file.name,
                "output_file": out_file.name,
                "num_packets": int(csi_clean.shape[0]),
                "num_subcarriers": int(csi_clean.shape[1]),
                "activity_code": activity_code,
                "activity_name": activity_name,
                "timestamp": "_".join(pcap_file.stem.split("_")[1:5]),
                "heart_rate_file": watch_file if watch_file else None
            }

            # Save meta.json
            meta_file = out_dir / (pcap_file.stem + "_meta.json")
            with open(meta_file, "w") as f:
                json.dump(meta, f, indent=4, ensure_ascii=False)

            print(f"[OK] {pcap_file.name} → {out_file.name}, meta.json written (HR: {watch_file})")

        except Exception as e:
            print(f"[ERROR] Failed to parse {pcap_file.name}: {e}")
