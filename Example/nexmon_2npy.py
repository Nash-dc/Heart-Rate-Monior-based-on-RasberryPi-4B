#!/usr/bin/env python3
import numpy as np
from nexcsi import decoder

pcap_file = "68.pcap"

print("[DEBUG] Starting parse:", pcap_file)

samples = decoder("raspberrypi").read_pcap(pcap_file)
print(samples.dtype.names)
raw_csi = samples['csi']

csi_full = decoder("raspberrypi").unpack(raw_csi)

null_idx = csi_full.dtype.metadata['nulls']
pilot_idx = csi_full.dtype.metadata['pilots']

csi_clean = np.delete(csi_full, null_idx, axis=1)
csi_clean = np.delete(csi_clean, pilot_idx, axis=1)

print("[+] Full CSI shape:", csi_full.shape) 
print("[+] Clean CSI shape:", csi_clean.shape) 
print("[+] First row (first 10 subcarriers):", csi_clean[0, :10])

np.save("csi_234.npy", csi_clean)
print("[+] Saved to csi_234.npy")
