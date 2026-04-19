# Self-Pruning Neural Network with Differentiable Gating (CIFAR-10)

## Overview

This project implements a self-pruning neural network that learns to remove unnecessary connections during training using a differentiable gating mechanism. Unlike traditional pruning methods applied after training, this approach integrates pruning directly into the optimization process, enabling the model to dynamically learn sparse representations.

The objective is to achieve high sparsity (model compression) while maintaining competitive accuracy.

---

## Motivation

Modern deep neural networks are highly over-parameterized, leading to inefficiencies in storage and computation. Traditional pruning techniques are typically applied after training, which can be suboptimal.

This project explores **in-training pruning**, where the model learns to identify and suppress redundant connections during training, resulting in:

* Reduced model size
* Improved computational efficiency
* Insight into parameter redundancy

---

## Core Idea

Each weight **w** is paired with a learnable gate **g**:

Effective Weight = w × g

Where:

- g = sigmoid(gate_scores)
- g ∈ (0, 1)

If g is small, the corresponding connection contributes very little and is effectively pruned.
---

## Architecture

* Custom Layer: **PrunableLinear**
* Network: Fully connected neural network
* Dataset: CIFAR-10 (subset: 5000 training / 1000 testing samples)
* Framework: PyTorch

---

## Loss Function

[
\text{Total Loss} = \text{CrossEntropyLoss} + \lambda \cdot | w \times g |_1
]

### Key Design Choice

Instead of penalizing gates directly, L1 regularization is applied to **effective weights (w × g)**.

This ensures that:

* The model cannot bypass sparsity by scaling weights
* True contribution of each connection is minimized

---

## Training Strategy

### Lambda Scheduling (Key Contribution)

A fixed λ was found to be unstable. Therefore, a scheduling strategy was introduced:

1. **Warm-up Phase**

   * λ = 0
   * Model learns features without pruning pressure

2. **Ramp Phase**

   * Gradual increase of λ
   * Controlled introduction of sparsity

3. **Full Phase**

   * λ reaches target value
   * Strong pruning enforced

This approach stabilizes training and enables effective pruning.

---

## Final Results

| λ (Scheduled) | Accuracy   | Weight Sparsity | Neuron Sparsity |
| ------------- | ---------- | --------------- | --------------- |
| 0.5           | 42.70%     | 99.93%          | 98.90%          |
| **1.5**       | **43.00%** | **99.93%**      | **98.90%**      |
| 4.0           | 41.90%     | 99.93%          | 98.90%          |

---

## Best Configuration

**λ = 1.5 (with scheduling)**

* Accuracy: **43.0%**
* Weight Sparsity: **99.93%**
* Neuron Sparsity: **98.90%**

This represents the optimal balance between performance and compression.

---

## Key Observations

### 1. Extreme Sparsity with Stable Accuracy

The model achieves nearly complete pruning (~99.9%) while maintaining accuracy (~43%), indicating significant redundancy in the network.

---

### 2. Sparsity–Accuracy Trade-off

* Low λ → insufficient pruning
* High λ → performance degradation
* Intermediate λ → optimal balance

---

### 3. Pruning Behavior

* Gates do not collapse to zero
* Mean gate values remain around ~0.2
* Pruning occurs through **effective weight suppression (w × g)**

---

### 4. Layer-wise Behavior

* Hidden layers: almost entirely pruned
* Output layer: remains active

This suggests that:

* Most intermediate representations are redundant
* Final layer retains critical decision boundaries

 ### 5. Lambda Scheduling
 Lambda scheduling was necessary for stable pruning;
 fixed lambda values resulted in either no sparsity or model collapse.


---

## Challenges and Debugging

### 1. No Sparsity with Initial Approach

Penalizing gates directly failed due to model compensation (weight scaling).

### 2. Incorrect Sparsity Metric

Using gate values was misleading. Corrected by evaluating effective weights.

### 3. Instability with Fixed λ

Fixed λ values caused either no pruning or model collapse.

### 4. Solution: Lambda Scheduling

Gradual application of sparsity pressure enabled stable and effective pruning.

---

## Design Choices

* Sigmoid-based differentiable gating
* L1 regularization on effective weights
* Cosine learning rate scheduler
* Lambda scheduling for stability
* Evaluation using effective weight threshold

---

## Limitations

* Unstructured pruning (not hardware-optimized)
* Gates do not fully collapse to zero
* Limited dataset size
* No post-pruning fine-tuning

---

## Future Work

* Structured pruning (neuron/channel level)
* Hard gating mechanisms
* Post-pruning retraining
* Extension to convolutional architectures
* Training on full CIFAR-10 dataset

---

## Key Takeaways

* Effective pruning requires penalizing true contribution (w × g)
* Scheduling is critical for stable sparsity learning
* Neural networks can tolerate extreme compression
* Most parameters in dense networks are redundant

---

## How to Run

```bash
python main.py
```

---

## Project Structure

```
self_pruning_nn/
│
├── src/
│   ├── layers.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
│
├── main.py
├── README.md
└── requirements.txt
```

---

## Author

Aryaneel Bhaduri
