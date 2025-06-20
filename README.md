# Fill-in-the-Blank Word Prediction using Bidirectional LSTM Networks

This repository implements a fill-in-the-blank prediction system using bidirectional LSTM networks with attention mechanisms. The model predicts missing words from context using both forward and backward LSTM pathways, and integrates confidence-based prediction selection for improved accuracy.

---

## 📚 Project Description

**Objective**: Predict missing words in sentences by leveraging context from both before and after the blank using LSTM models.

- **Forward LSTM**: Predicts the blank word using context from the start of the sentence up to the blank.
- **Backward LSTM**: Predicts using context from after the blank, in reversed order.
- **Prediction Selection**: A confidence-based mechanism is used to combine the outputs of both models.

---

## 📦 Dataset

- **Dataset Used**: [RACE Dataset](https://huggingface.co/datasets/race)
- **Source**: English reading comprehension passages and questions.
- **Preprocessing Steps**:
  - Filter out short or low-quality sentences
  - Convert text to lowercase
  - Remove non-alphanumeric tokens
  - Exclude function words from blank candidates
  - Create forward and backward input sequences
  - Tokenize using BERT tokenizer (max length = 50)

To load:
```python
from datasets import load_dataset
dataset = load_dataset("race", "all")
🧠 Model Architecture
🔹 Embedding Layer
Input dim: vocab_size

Output dim: 256

Dropout: 0.3

🔹 Bidirectional LSTM
2 layers

Hidden dimension: 512

Dropout: 0.3

Batch-first

Bidirectional: True

🔹 Attention Mechanism
4 attention heads

Dropout: 0.1

Separate attention for forward and backward paths

🔹 Output Layers
Linear → LayerNorm → ReLU → Dropout (0.2) → Final Linear Layer

Confidence Scoring Module (dense layers + sigmoid)

⚙️ Training Details
Batch Size: 16

Learning Rate: 1e-4

Epochs: 5

Optimizer: AdamW

Scheduler: ReduceLROnPlateau (patience=2, factor=0.5)

Regularization: L2 (λ = 0.01), Dropout, Label smoothing

Gradient Clipping: max_norm = 1.0

Early Stopping: patience = 3

🔬 Evaluation and Analysis
Directional LSTM Performance
Forward LSTM:

Effective at modeling grammatical structure

Performs well on functionally structured sentences

Backward LSTM:

Captures long-range dependencies from future context

Better with narrative and complex sentence endings

Prediction Fusion Strategy
Temperature scaling (T=0.7)

Confidence-weighted averaging of forward/backward predictions

🧪 Implementation Highlights
🔹 Robust text cleaning and filtering pipeline

🔹 Dual LSTM models for context-aware predictions

🔹 Attention mechanism for enhanced focus

🔹 Confidence scoring for smart prediction selection

🔹 Handles variable-length input using padding and BERT tokenizer

🧠 Observations
Combining both models significantly improved prediction reliability.

Confidence-weighted averaging helps resolve conflicts between LSTM outputs.

Strategic data filtering greatly improved model performance and convergence.

Performance decreased on ambiguous sentences or complex grammar cases.


🏁 Conclusion
This project demonstrates that bidirectional LSTM models, paired with attention and confidence scoring, can effectively handle the fill-in-the-blank task on unstructured language data. By balancing context from both directions, we can achieve a robust prediction system for natural language understanding tasks.

