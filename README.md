# 🎧 Audio Forensics & Digital Tampering Detection

![Banner](./assets/banner.png)

## 📑 Table of Contents

- 🧠 [Project Overview](#project-overview)
- 🎯 [Core Objectives](#core-objectives)
- 🛠️ [System Architecture](#system-architecture)
- 🧪 [Training Workflow](#training-workflow)
- 📊 [Evaluation & Metrics](#evaluation--metrics)
- 🔥 [Bias Handling](#bias-handling)
- ⚙️ [Setup & Installation](#setup--installation)
- 🙏 [Acknowledgments](#acknowledgments)
- 💼 [Libraries & Tools](#libraries--tools)
- 🤝 [Contact & Contribution](#contact--contribution)
- 📜 [License](#license)

---

## 🧠 Project Overview

In the digital age, audio tampering is a critical threat to legal systems, journalism, and public trust. Our project provides a reliable solution to **detect forged audio**, **transcribe conversations**, and **analyze speaker emotions** to aid in digital forensics and investigation.

This system:
- Detects **audio tampering**
- Performs **speech transcription** using **Whisper**
- Analyzes **voice emotion** as positive, negative, or neutral

---

## 🎯 Core Objectives

Our main goals are:

- ✅ Detect and classify tampered vs untampered audio
- 🎙️ Transcribe spoken content using OpenAI's **Whisper**
- 💬 Perform **audio sentiment analysis** to classify emotion
- 🧪 Ensure scalability, performance, and forensic-grade reliability
- 🔐 Empower cyber and digital forensic professionals with intelligent tools

---

## 🛠️ System Architecture

Our system architecture involves an end-to-end pipeline built with modern tools and clear separation of responsibilities:

```text
📤 Upload Audio File (React.js Frontend)
↓
🔊 Preprocess Audio (pydub, librosa, scipy)
↓
🧠 Tampering Detection (Random Forest Classifier)
↓
🗣 Transcription (Whisper ASR)
↓
😊 Sentiment Analysis (SVM/ML Classifier)
↓
📊 JSON Response Displayed on Frontend
## 🧪 Training Workflow

### 1. Dataset Collection
We collected audio samples from sources like **ASVspoof**, **Kaggle**, and our own recordings. Two primary datasets were used:
- **Tampered vs Original** for forgery detection
- **Positive, Negative, Neutral** for sentiment analysis

### 2. Feature Extraction
We used **MFCC (Mel-Frequency Cepstral Coefficients)** via `librosa` to extract meaningful acoustic patterns.

```python
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
features = np.mean(mfcc.T, axis=0)
