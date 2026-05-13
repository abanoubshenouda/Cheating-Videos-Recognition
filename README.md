# Cheating Videos Recognition in Exam Halls

An end-to-end Deep Learning project for detecting cheating behavior in exam hall videos. The system collects video data, preprocesses it into frames and temporal sequences, trains multiple video-recognition models, and triggers a notification when cheating behavior is detected.

The project is designed as a team-based research pipeline where each member experiments with a different model family, then the results can be compared or fused using an ensemble/fusion layer.

![Project Architecture](project%20structure.jpeg)

## Table of Contents

- [Project Objective](#project-objective)
- [Main Pipeline](#main-pipeline)
- [Team Model Tracks](#team-model-tracks)
- [Cheating Actions](#cheating-actions)
- [Dataset Collection](#dataset-collection)
- [Preprocessing Output Structure](#preprocessing-output-structure)
- [Current Repository Structure](#current-repository-structure)
- [Model Inputs and Outputs](#model-inputs-and-outputs)
- [Current Baseline Result](#current-baseline-result)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Real-Time Notification Logic](#real-time-notification-logic)
- [Recommended Evaluation Metrics](#recommended-evaluation-metrics)
- [Future Improvements](#future-improvements)

## Project Objective

The goal is to build a video recognition system that detects suspicious cheating actions during exams. The dataset contains short videos or video segments showing normal exam behavior and possible cheating behaviors.

The final system should:

- collect extra video samples using the Pexels API;
- preprocess raw videos into frames and frame sequences;
- train different deep neural network models;
- classify behavior as `cheating` or `normal`;
- return confidence scores;
- generate a real-time notification when cheating is detected.

## Main Pipeline

```text
Raw Video
   |
   v
Frame Extraction
   |
   v
Preprocessing + Labeling
   |
   v
Feature Extraction / Sequence Modeling
   |
   v
Action Classification
   |
   v
Cheating Detection
   |
   v
Real-Time Notification
```

The full project flow is:

```text
Pexels API Scraping
   -> Raw Videos
   -> Master Preprocessing Notebook
   -> Frames + Sequences
   -> Individual Model Training
   -> Model Comparison / Fusion
   -> Cheating Detection
   -> Alert + Timestamp + Frame Snapshot
```

## Team Model Tracks

| Member | Model Track | Main Idea | Expected Input |
|---|---|---|---|
| Nour Hany | TimeSformer | Video transformer using space-time attention | Frame sequences / video clips |
| Abanoub Shenouda | CNNs + Transfer Learning ResNet/VGG | Image-level feature extraction and binary classification | Individual frames |
| George Fakhoury | RNN / LSTM Sequence Modeling | Temporal modeling over extracted frame features | Frame sequences |
| Paula Adel | Seq2Seq VAE + GRU Generative | Sequence representation learning and generative modeling | Frame sequences |
| Maria Ayman | Transformers Attention ViT | Vision Transformer attention over image patches | Individual frames or sampled clips |

The project can evaluate each model independently, then compare results or combine them using:

- majority voting;
- weighted voting;
- feature-level fusion;
- confidence-score fusion.

## Cheating Actions

The dataset should cover both normal and suspicious exam behaviors.

Suggested cheating action categories:

| Action Class | Description | Final Binary Label |
|---|---|---|
| `phone_use` | Student uses or hides a phone during the exam | `cheating` |
| `looking_sideways` | Student repeatedly looks at another paper or student | `cheating` |
| `whispering` | Students communicate secretly during the exam | `cheating` |
| `passing_paper` | Student passes a note or paper | `cheating` |
| `hiding_notes` | Student reads or hides cheat notes | `cheating` |
| `normal_behavior` | Writing, reading, focusing, or sitting normally | `normal` |

For the current binary task, cheating-related actions can be merged into one label:

```text
phone_use, looking_sideways, whispering, passing_paper, hiding_notes -> cheating
normal_behavior -> normal
```

## Dataset Collection

The project includes scripts for collecting extra videos using the Pexels API:

| Script | Purpose |
|---|---|
| `pexels_scraper.py` | Searches Pexels by action-specific queries, downloads videos, extracts frames, saves metadata, and creates train/val/test splits |
| `download_videos.py` | Downloads raw videos from Pexels into a dataset folder using predefined search queries |

### Pexels API Setup

1. Create a free API account from [Pexels API](https://www.pexels.com/api/).
2. Add your API key in the scraper script.
3. Use action-based search queries such as:

```text
student using phone exam
student looking sideways classroom
students whispering classroom
student passing note paper
student hiding cheat sheet
student writing exam paper
```

### Important Data Collection Notes

- Pexels videos may not perfectly match real exam cheating behavior, so manual review is required.
- Remove irrelevant or misleading videos before training.
- Keep metadata for each video: source, query, class, duration, and download date.
- Avoid mixing frames from the same video across train, validation, and test sets.
- Respect Pexels API terms and attribution requirements.
- Do not expose API keys in public repositories.

## Preprocessing Output Structure

The master preprocessing notebook creates the following shared structure:

```text
processed_data/
├── frames/
│   ├── train/
│   │   ├── normal/
│   │   └── cheating/
│   ├── val/
│   │   ├── normal/
│   │   └── cheating/
│   └── test/
│       ├── normal/
│       └── cheating/
└── sequences/
    ├── train/
    │   ├── normal/
    │   └── cheating/
    ├── val/
    │   ├── normal/
    │   └── cheating/
    └── test/
        ├── normal/
        └── cheating/
```

### What the Master Preprocessing Does

`Master_Preprocessing.ipynb` performs:

- raw video loading;
- train/validation/test splitting;
- frame extraction;
- frame resizing to `224x224`;
- padding to preserve aspect ratio;
- sequence creation using fixed-length clips;
- saving processed frames and sequences in a shared format.

Current preprocessing settings:

| Setting | Value |
|---|---:|
| Image size | `224 x 224` |
| Sample FPS | `6` |
| Sequence length | `16` frames |
| Sequence stride | `8` frames |
| Output frame format | `.jpg` |
| Output sequence format | `.npy` |

## Current Repository Structure

```text
.
├── Master_Preprocessing.ipynb
├── Abanoub_CNN_VGG16.ipynb
├── camera_test_colab.ipynb
├── camera_test_with_display.ipynb
├── pexels_scraper.py
├── download_videos.py
├── project structure.jpeg
├── processed_data/
│   ├── frames/
│   └── sequences/
├── abanoub_cheating_detection_results/
│   ├── best_model.keras
│   ├── training_curves.png
│   └── confusion_matrix.png
├── PROJECT_REPORT.md
└── Cheating_Videos_Recognition_Report.docx
```

## Model Inputs and Outputs

### Frame-Based Models

Used by CNNs, ResNet/VGG, and ViT-style models.

```text
Input:  224 x 224 x 3 RGB frame
Output: cheating / normal + confidence score
```

### Sequence-Based Models

Used by TimeSformer, LSTM/GRU, Seq2Seq, and video transformer models.

```text
Input:  sequence of frames, usually 16 frames per clip
Output: cheating / normal + confidence score
```

### Notification Output

When cheating is detected, the system should return:

```text
label: cheating
confidence: model probability
timestamp: detection time
snapshot: frame where cheating was detected
alert: notification message
```

## Current Baseline Result

The current baseline is a VGG16 transfer learning model trained by Abanoub Shenouda.

Saved results:

```text
abanoub_cheating_detection_results/
├── best_model.keras
├── training_curves.png
└── confusion_matrix.png
```

Frame-level test accuracy calculated from the saved confusion matrix:

```text
78.52%
```

Training curves:

![Training Curves](abanoub_cheating_detection_results/training_curves.png)

Confusion matrix:

![Confusion Matrix](abanoub_cheating_detection_results/confusion_matrix.png)

## Installation

Recommended Python version:

```text
Python 3.10+
```

Install the main dependencies:

```bash
pip install tensorflow opencv-python numpy scikit-learn matplotlib seaborn pillow requests ipywidgets
```

For PyTorch-based models such as TimeSformer, ViT, or LSTM pipelines:

```bash
pip install torch torchvision torchaudio
```

For transformer-based experimentation:

```bash
pip install transformers timm einops
```

## How to Run

### 1. Collect Videos from Pexels

Edit the API key and queries in the scraper script, then run:

```bash
python pexels_scraper.py
```

or use the simpler downloader:

```bash
python download_videos.py
```

Expected raw dataset idea:

```text
dataset/
├── normal/
└── cheating/
```

If using multi-action folders from the scraper, review the videos manually and map cheating actions into the final binary labels before running the master preprocessing step.

### 2. Preprocess the Dataset

Open and run:

```text
Master_Preprocessing.ipynb
```

This generates:

```text
processed_data/frames
processed_data/sequences
```

### 3. Train the Baseline CNN/VGG Model

Open and run:

```text
Abanoub_CNN_VGG16.ipynb
```

This trains the VGG16 transfer learning model and saves the best model.

### 4. Train Other Team Models

Each member should use the shared `processed_data` folder:

- frame-based models should use `processed_data/frames`;
- sequence-based models should use `processed_data/sequences`.

### 5. Run Live Camera Testing

For Google Colab:

```text
camera_test_colab.ipynb
```

For display with frame overlay:

```text
camera_test_with_display.ipynb
```

Main camera settings:

```python
TEST_DURATION = 35
CHEAT_THRESHOLD = 0.5
IMG_SIZE = (224, 224)
```

## Real-Time Notification Logic

The live system classifies multiple frames during a short time window.

Example logic:

```python
if cheating_frames / total_frames > CHEAT_THRESHOLD:
    final_result = "CHEATING"
else:
    final_result = "NORMAL"
```

A good notification should include:

- predicted class;
- confidence score;
- timestamp;
- suspicious frame snapshot;
- optional sound or UI alert.

## Recommended Evaluation Metrics

Accuracy alone is not enough because the dataset is imbalanced. The project should report:

- accuracy;
- precision;
- recall;
- F1-score;
- confusion matrix;
- ROC-AUC if probability scores are available;
- false positive rate for normal students;
- false negative rate for missed cheating cases.

For this project, recall on the `cheating` class is especially important because missing cheating cases is a critical failure.

## Future Improvements

- Add more real exam-like videos to reduce domain mismatch.
- Balance the dataset by collecting more `normal_behavior` examples.
- Use manual annotation to remove noisy or irrelevant Pexels videos.
- Train a multi-class model to detect the cheating type, not only binary cheating/no-cheating.
- Compare frame-based and sequence-based models using the same train/val/test split.
- Add ensemble voting between the five model tracks.
- Optimize the final model for real-time inference.
- Save alert snapshots and timestamps in a structured log file.

## Full Report

The full project report is available here:

```text
PROJECT_REPORT.md
```

The Word version is also available:

```text
Cheating_Videos_Recognition_Report.docx
```
