# Pulse-Fi: Contactless Heart Rate Monitoring with Wi-Fi CSI

This project reproduces **Pulse-Fi** (*published on August 12, 2025*),  
a low-cost system for accurate, non-intrusive heart rate monitoring using  
Wi-Fi Channel State Information (CSI).

We implemented both the **signal processing pipeline** and the **lightweight LSTM regression model**,  
and successfully reproduced the results reported in the paper.  
We also set up the **same hardware environment (Wi-Fi CSI collection with ESP32 / Nexmon)**  
and obtained consistent outcomes.

---

## 🔬 Background

Traditional heart rate monitoring often relies on wearables (smartwatches, chest straps)  
or cameras (PPG), which may be inconvenient or raise privacy concerns.  

**Pulse-Fi** demonstrates that commodity Wi-Fi devices can capture subtle chest movements  
caused by heartbeats. By processing the amplitude of CSI signals, applying band-pass  
filters, smoothing, and feeding them into an LSTM model, we can estimate heart rate  
with high accuracy (error < 0.5 BPM in some cases).

---

## 📂 Project Structure

```
.
├── bash_script/              # Shell scripts for automation
├── CleanData/                # Preprocessed CSI data (ignored by git)
├── Model/                    # Saved models (e.g., best_model.pt)
├── TestCase/                 # Test cases for validation
├── model_evaluate_script/    # Evaluation pipeline (e.g., eval_pulsefi.py)
├── train_script/             # Training pipeline (e.g., train_pulsefi.py)
├── predict_hr.py             # Predict HR from smartwatch JSON data
├── predict_pulsefi.py        # Predict HR from CSI using trained LSTM model
├── cli_verify5.py            # Command-line verification utility
├── nexmon_2npy.py            # Convert raw Nexmon PCAP CSI → npy format
├── Evaluate.png              # Visualization of prediction vs ground truth
├── UsefulCommands.txt        # Handy CLI commands and notes
├── .gitignore                # Ignore rules (e.g., CleanData/)
└── README.md                 # Project documentation (this file)
```

---

## ⚙️ Installation

Requirements:
- Python ≥ 3.8
- [PyTorch](https://pytorch.org/)
- NumPy, SciPy, Matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🏃 Usage

### 1. Convert raw CSI (Nexmon → npy)
```bash
python nexmon_2npy.py --input raw.pcap --output CleanData/session.npy
```

### 2. Train the model
```bash
python train_script/train_pulsefi.py     --csi-dir CleanData     --watch-dir Data_DS1_smartwatch/Data     --output Model/pulsefi_model.pt     --window-sec 20     --hop-sec 1
```

### 3. Evaluate the model
```bash
python model_evaluate_script/eval_pulsefi.py     --csi-dir CleanData     --watch-dir Data_DS1_smartwatch/Data     --model Model/pulsefi_model.pt     --hidden-dim 128     --num-layers 1
```

### 4. Predict heart rate on new CSI
```bash
python predict_pulsefi.py --input CleanData/new_session.npy --model Model/pulsefi_model.pt
```

---

## 📊 Results

- With 20–30s sliding windows, the model achieves **MAE ≈ 0.2–0.4 BPM**.  
- Both software and hardware setups were replicated, confirming feasibility on commodity Wi-Fi devices.  

![Evaluation](Evaluate.png)

---

## 🔮 Future Work

- Integration with **clinical applications** (e.g., breast cancer recovery monitoring)  
- HRV estimation and psychological well-being assessment  
- Embedded deployment (ESP32 / Raspberry Pi)  

---

## 📚 Reference

- **Pulse-Fi: A Low-Cost System for Accurate Heart Rate Monitoring Using Wi-Fi Channel State Information**,  
  published on August 12, 2025.