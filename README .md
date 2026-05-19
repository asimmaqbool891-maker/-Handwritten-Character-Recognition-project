# ✏️ Handwritten Character Recognition (HCR)

**TASK 3 — Deep Learning Project**

---

## 📌 Overview

A complete end-to-end system that identifies handwritten characters (digits 0–9 and letters A–Z / a–z) using a **Convolutional Neural Network (CNN)** trained on the **MNIST** and **EMNIST Balanced** datasets.

---

## 🗂️ Project Structure

```
handwriting_recognition/
│
├── train_model.py        # ← CNN model training (MNIST + EMNIST)
├── predict.py            # ← Predict on image files / demo
├── app.py                # ← Streamlit web app (interactive UI)
├── requirements.txt      # ← Python dependencies
├── README.md             # ← This file
│
└── saved_model/          # ← Created after training
    ├── handwriting_cnn.keras   # Trained model weights
    ├── class_labels.pkl        # Label mapping list
    ├── training_history.pkl    # Loss/accuracy history
    └── training_curves.png     # Training plot
```

---

## 📦 Datasets

| Dataset | Content | Classes |
|---------|---------|---------|
| **MNIST** | Handwritten digits 0–9 | 10 |
| **EMNIST Balanced** | Digits + uppercase + some lowercase | 47 |

- MNIST: 60,000 training / 10,000 test images (28×28 grayscale)
- EMNIST Balanced: 112,800 training / 18,800 test images (28×28 grayscale)
- Both are downloaded **automatically** on first run via Keras / TensorFlow Datasets.

---

## 🧠 CNN Architecture

```
Input (28×28×1)
│
├── Conv Block 1: Conv2D(32) → BN → Conv2D(32) → MaxPool → Dropout(0.25)
├── Conv Block 2: Conv2D(64) → BN → Conv2D(64) → MaxPool → Dropout(0.25)
├── Conv Block 3: Conv2D(128) → BN → MaxPool → Dropout(0.25)
│
├── Flatten
├── Dense(256) → BN → Dropout(0.5)
└── Dense(47, softmax)   ← 47 character classes
```

**Techniques used:**
- Batch Normalization (stable, faster training)
- Dropout (prevents overfitting)
- Data Augmentation (rotation, zoom, shift)
- Learning Rate Reduction on Plateau
- Early Stopping with best-weight restoration

---

## ⚙️ Setup & Installation

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```
Training takes ~5–20 min depending on your hardware (GPU recommended).

### 3. Run predictions

**On a custom image:**
```bash
python predict.py path/to/your/image.png
```

**Quick demo (random MNIST samples):**
```bash
python predict.py
```

### 4. Launch web app
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

---

## 📊 Expected Results

| Dataset | Accuracy |
|---------|----------|
| MNIST only | ~99.2% |
| MNIST + EMNIST Balanced | ~87–90% |

---

## 🔄 Extendability (CRNN for Words/Sentences)

As mentioned in the task, this system can be extended to full **word/sentence recognition** using a **CRNN (Convolutional Recurrent Neural Network)**:

```
Image → CNN (feature extraction) → RNN/LSTM (sequence modeling) → CTC Loss → Text output
```

The CNN backbone from this project can serve directly as the feature extractor in a CRNN pipeline.

---

## 📁 Output Files

After training:
- `saved_model/handwriting_cnn.keras` — Full trained model
- `saved_model/class_labels.pkl` — Class label list
- `saved_model/training_curves.png` — Accuracy & loss plots
- `prediction_result.png` — Saved after each prediction
- `demo_results.png` — Saved after running demo mode
