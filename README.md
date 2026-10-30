# 🌿 Plant Leaf Disease Classifier

AI-powered plant leaf disease detection built on **MobileNetV2 transfer learning**.  
Upload a tomato leaf photo and get an instant health diagnosis with confidence scores and expert agricultural recommendations.

> Built during the AI/ML Internship at Taviq AI Solutions.

## ✨ Features
- Transfer Learning with **MobileNetV2** (ImageNet pre-trained weights).
- Custom confidence-threshold fallback for low-confidence predictions.
- Top-3 prediction confidence matrix with treatment recommendations.
- Clean, responsive Streamlit UI.
- Dataset pipeline with 70/15/15 train–val–test split (`src/dataset.py`).

## 🚀 Quick Start

### 1. Activate the virtual environment
```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run the app
```powershell
streamlit run app.py
```
Then open the URL shown in the terminal (default: **http://localhost:8501**).

### Notes
- Pretrained PyTorch weights are cached on the **D drive** (`D:/pytorch_cache`) so the system drive is not filled up. This is set automatically in `app.py` and `src/model.py` via the `TORCH_HOME` environment variable.
- If no trained checkpoint exists at `models/best_model.pth`, the app runs in **initialization mode** (random weights) and displays a notice. Train and save your weights to `models/best_model.pth` for real predictions.

## 📁 Project Structure
```
app.py                  # Streamlit application entry point
requirements.txt        # Python dependencies
README.md               # This file
assets/
  disease_tips.json     # Per-disease treatment recommendations
src/
  model.py              # Model builders (MobileNetV2 transfer + baseline CNN)
  dataset.py            # PlantVillage dataset + 70/15/15 data loaders
models/                 # Trained checkpoints go here (best_model.pth)
```

## 🛠 Tech Stack
- Python 3.11 · PyTorch · TorchVision · Streamlit · Pillow · NumPy · scikit-learn

## ⚠️ Disclaimer
Demonstration model only. For serious agricultural decisions, consult a qualified plant pathologist or agronomist.
