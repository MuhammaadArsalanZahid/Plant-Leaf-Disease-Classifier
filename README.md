# 🌿 Plant Leaf Disease Classifier (Deep Learning Architecture)

An end-to-end deep learning image classification system built during the AI/ML Internship at **Taviq AI Solutions**. This application utilizes two-phase transfer learning on a convolutional neural network to automatically diagnose crop diseases from plant leaf photographs, evaluate model certainty via strict safety thresholds, and offer structural agricultural remediation strategies.

---

## 📺 Project Demo
*   **Demo Video Link:** https://drive.google.com/file/d/14Ts8g-g1W2HF7XQ0A4TZRsfodKYG3ero/view?usp=sharing
*   **LinkedIn Project Post:** [Insert Your LinkedIn Project Post Link Here]

---

## 🛠️ System Architecture & Core Features

The pipeline is engineered to ensure data integrity and provide safe, actionable predictions in production environments:

```text
User Image ──► Preprocessing (224x224 Normalized Tensor) ──► Pretrained MobileNetV2
                                                                     │
   ┌─────────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────┐
   ▼                                                                                                                                   ▼
[Top Confidence < 65%] ──► Trigger Safety Guard                                                      [Top Confidence ≥ 65%] ──► Accept Diagnosis
   │                                                                                                                                   │
   ▼                                                                                                                                   ▼
"Uncertain: Please provide a clearer photo."                                                        Render Top 3 Bar Charts + Json Remediation Tips
```

### Key Engineering Implementations:
1. **Isolated Data Segregation:** Structured a deterministic pipeline splitting the dataset into exact 70% Train, 15% Validation, and 15% Test sets. Random seeds were locked to guarantee that no training images leaked into validation frames.
2. **Asymmetric Regularization & Augmentation:** Injected random horizontal flips, bounding rotations (±15°), color jitter, and zoom parameters explicitly into the training graph via `src/dataset.py`. Validation and test partitions remained completely unaugmented.
3. **Two-Phase Fine-Tuning Execution (`train.py`):** 
   * **Phase 1:** Froze the pretrained feature layers of a `MobileNetV2` network and trained an appended linear classification head with a Dropout layer (0.3) at a learning rate of 10⁻³ to minimize target cross-entropy loss.
   * **Phase 2:** Unfroze the deep convolutional blocks and applied a micro-learning rate of 10⁻⁵ to perform fine-grained parameter tuning while preserving structural feature extractors.
4. **Production-Safe Inference Thresholding:** Integrated an analytical confidence threshold constraint (65%) into `app.py`. If an out-of-domain photograph (such as a non-leaf file or a noisy background) is provided, the system overrides the softmax distribution and requests a cleaner specimen.
5. **Decoupled Asset Guidance:** Configured the application frontend to fetch treatment strategies from a clean, decoupled `assets/disease_tips.json` directory. This links automated agronomy advice directly to the targeted prediction index.

---

## 🗂️ Exact Repository Structure

Based on our active production workspace, the directory is laid out as follows:

```text
plant-leaf-disease-classifier/
├── app.py                 # Streamlit web application deployment frontend
├── train.py               # Complete two-phase training and evaluation execution pipeline
├── requirements.txt       # Unified library dependency declarations
├── README.md              # Project documentation and engineering guide
├── app_err.txt            # Local development runtime error tracking logs
├── app_out.txt            # Standard system output and execution tracking logs
├── assets/
│   └── disease_tips.json  # Metadata dictionary containing disease treatment guidelines
├── dataset/               # PlantVillage crop image sub-directories (Train/Val/Test slices)
├── models/                # Local serialized checkpoints (.pth model state dictionaries)
└── src/
    ├── __init__.py
    ├── dataset.py         # Transforms mapping, splitting logic, and custom DataLoaders
    └── model.py           # Custom Baseline CNN and Transfer Learning model architectures
```

---

## 🚀 Installation & Local Deployment

Follow these sequential setup instructions to spin up the application locally:

### 1. Activate the Workspace Environment
Open a terminal inside your repository path and execute the activation script:
```powershell
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
```
*(Once activated, you will see `(venv)` prepended to your command line).*

### 2. Configure Local Space Path Redirections
To handle Python weights caching seamlessly on your secondary storage partition, link your environment path markers directly to your D-Drive:
```powershell
\$env:TEMP="D:\pip_temp"
\$env:TMP="D:\pip_temp"
\$env:TORCH_HOME="D:\pytorch_cache"
```

### 3. Install Requirements
Execute pip mapping using the official lightweight CPU index URL to bypass heavy unnecessary GPU assets:
```powershell
# Install optimized PyTorch stack
pip install torch torchvision --index-url https://pytorch.org --cache-dir="D:\pip_temp"

# Deploy standard components
pip install -r requirements.txt --cache-dir="D:\pip_temp"
```

### 4. Run the Training Script
To execute or verify your deep learning training loop, loss values, and save your model state to the `models/` directory, run:
```powershell
python train.py
```

### 5. Launch the Web Interface
Execute the frontend dashboard engine:
```powershell
streamlit run app.py
```
Open your web browser and navigate to `http://localhost:8501`.

---

## 📊 Evaluation & Empirical Performance
The final transfer learning network was evaluated against the test partition, showing significant optimization gains over the custom baseline model.

| Model Architecture Tested | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| Custom Baseline CNN | 62.4% | 60.1% | 59.8% | 59.9% |
| **MobileNetV2 (Transfer Learning)** | **94.7%** | **94.2%** | **94.5%** | **94.3%** |

---

## ✒️ Identity & Acknowledgments
* **Developer Name:** Muhammad Arsalan Zahid
* **LinkedIn Profile:** https://www.linkedin.com/in/muhammad-arsalan-zahid-aa255b330/
* *Built during the AI/ML Internship at Taviq AI Solutions.*
