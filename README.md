# Facial Emotion Recognition Using Deep Learning (CNN)

---

## Project Overview

Convolutional Neural Network that classifies human facial expressions into 7 emotion categories from 48x48 grayscale images in real time. Trained on the FER2013 benchmark dataset and deployed via Gradio as a publicly accessible web interface.

**7 Emotion Classes:** Angry | Disgust | Fear | Happy | Sad | Surprise | Neutral

**Applications:** Mental health monitoring, customer satisfaction detection, driver fatigue alerting, adaptive educational platforms.

---

## Dataset — FER2013

| Property | Detail |
|---|---|
| Total images | 35,887 grayscale 48x48 face images |
| Train split | 28,709 images |
| Test split | 7,178 images |
| Image size | 48 x 48 pixels, 1 channel (grayscale) |
| Class with most samples | Happy (~8,989 images) |
| Class with fewest samples | Disgust (~547 images) |
| Key challenge | 16x class imbalance; Fear/Surprise visually similar |

---

## CNN Architecture

```
Input (48, 48, 1) → pixel/255 normalisation
Conv2D(64, 5x5) → BatchNorm → MaxPool(2x2) → Dropout(0.25)
Conv2D(128, 3x3) → BatchNorm
Conv2D(512, 3x3) → BatchNorm → MaxPool(2x2) → Dropout(0.25)
GlobalAveragePooling2D
Dense(256, LeakyReLU) → Dropout(0.5)
Dense(7, Softmax)
```

**Design choices:**
- BatchNorm after each Conv: accelerates convergence, reduces sensitivity to initialisation
- Global Average Pooling instead of Flatten+Dense: reduces parameters, prevents overfitting
- Dropout(0.5) before output: heaviest regularisation where memorisation risk is highest
- LeakyReLU: prevents dead neurons (allows small negative gradient for x < 0)

---

## Data Augmentation

Applied via `ImageDataGenerator` to prevent overfitting:

| Augmentation | Value |
|---|---|
| Rotation range | ±25 degrees |
| Zoom range | 20% |
| Horizontal flip | True |
| Brightness range | [0.8, 1.2] |

Each epoch sees a differently augmented version of every training image — forces the model to be invariant to lighting, angle, and scale variations.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=0.0001) |
| Loss | categorical_crossentropy |
| Batch size | 64 |
| Max epochs | 100 (EarlyStopping) |
| class_weight | Computed to compensate Disgust imbalance |

**Callbacks:**
- `ModelCheckpoint` — saves model only when `val_accuracy` improves
- `EarlyStopping` — patience=10 on `val_loss`; restores best weights
- `ReduceLROnPlateau` — halves LR when `val_loss` stagnates for 3 epochs

---

## Model Performance

| Emotion | Performance | Notes |
|---|---|---|
| Happy | Highest F1 | Most visually distinct (smile, raised cheeks) |
| Disgust | Lowest F1 | Fewest training samples (547 images) |
| Fear / Surprise | Most confused | Share raised brows, wide eyes, open mouth |
| Neutral | Good F1 | Absence of strong features |

Overall test accuracy is competitive with published FER2013 benchmarks (63–73%). Human accuracy on FER2013 is approximately 65%.

---

## Deployment — Gradio

```python
gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(type='pil', shape=(48, 48)),
    outputs=gr.Label(num_top_classes=3),
    share=True       # generates public URL valid 72 hours
)
```

- User uploads image → converted to grayscale 48x48 → normalised → `model.predict()`
- Output: top-3 predicted emotions with softmax probabilities
- No local Python installation required for end users

---

## Tech Stack

```
Python | TensorFlow / Keras | NumPy | Matplotlib | Scikit-learn | Gradio
CNN | BatchNorm | GlobalAveragePooling | ImageDataGenerator
```

---

## Repository Contents

```
Facial-Emotion-Recognition-Using-Deep-Learning/
├── Facial_expression_recognition_using_cnn (1).ipynb   # Main training notebook
├── Facial_Emotion_Recognition_Presentation.pptx        # Presentation (11 slides)
├── angry2.webp                                          # Sample test images
├── cry.webp
├── happy.webp / happy3.jpg / happy3.webp
├── neutral.webp
├── portrait-sad-man-600nw-126009806.png
└── surprise1.jpg
```

---

## How to Run

```bash
pip install tensorflow numpy matplotlib scikit-learn gradio
jupyter notebook "Facial_expression_recognition_using_cnn (1).ipynb"
```

---

*AlmaBetter M.Sc. Data Science | 2026 | Vishal Kumar Singh*
