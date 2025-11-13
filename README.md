# DistilBERT-1D-CNN-for-Multi-Class-Classification
This notebook implements a text classification pipeline using DistilBERT as the base language model, augmented with a 1D Convolutional Neural Network (CNN) and Global Max Pooling for sequence modeling
# DistilBERT-CNN-Reddit-Classifier

A TensorFlow-based text classification model combining **DistilBERT** and a lightweight **1D-CNN head** to classify Reddit posts into 6 categories.

## 🎯 Objective
Classify user posts from Reddit into one of 6 mental-health or emotion-related categories using deep transfer learning.

## 🔧 Architecture
- **Base Model**: `distilbert-base-uncased`
- **Custom Head**:
  - `Conv1D` (64 filters, kernel=5, ReLU)
  - `GlobalMaxPooling1D`
  - Dense (128, ReLU) → Output (6, Softmax)

## 📊 Dataset
- Source: `/kaggle/input/complete-reddit/`
- Files: `posts_train.csv`, `posts_val.csv`, `posts_test.csv`
- Label column: `class_id` (6 classes)

## 📈 Evaluation
- Training/validation loss & accuracy plots
- Per-class ROC curves & AUC
- Class-wise accuracy report (e.g., Class 5: 91.6%, Class 2: 72.9%)

## 🚀 Results (Validation Set)
| Class | Accuracy |
|-------|----------|
| 0     | 78.5%    |
| 1     | 74.8%    |
| 2     | 72.9%    |
| 3     | 78.9%    |
| 4     | 82.9%    |
| 5     | **91.6%**|

✅ **Best performance on Class 5** — suggests high separability or data balance.  
⚠️ **Class 2 is hardest** — may need more data or error analysis.

## 🛠️ Dependencies
- `tensorflow`, `transformers`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`

## 📌 Note
- ❗ **BiLSTM is *not used*** — only CNN + pooling is added after DistilBERT.
- All preprocessing and tokenization handled via Hugging Face `transformers`.
