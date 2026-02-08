# 🧠 LeNet-5: Regular vs Depthwise Separable Convolution Analysis

**Academic Project - Neural Network Efficiency Study**  
*88.24% operation reduction with only 2.08% accuracy drop on MNIST*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MNIST](https://img.shields.io/badge/Dataset-MNIST-green.svg)](http://yann.lecun.com/exdb/mnist/)

---



## 🎯 Problem Statement

### **Academic Assignment Context**
This project was developed as part of a machine learning systems course assignment to analyze computational efficiency in convolutional neural networks.

### **Specific Objectives:**
1. **Implement LeNet-5** using PyTorch and train on MNIST dataset
2. **Replace all convolution layers** with depthwise separable convolutions
3. **Calculate and compare** total multiply-add operations for both architectures
4. **Derive mathematical function** that maps regular convolution operations to depthwise separable convolution operations

### **Research Questions:**
- How much computation can be saved using depthwise separable convolutions?
- What is the mathematical relationship between the two convolution types?
- What is the accuracy-efficiency trade-off?
- Why are depthwise separable convolutions used in mobile architectures?

---

## 🛠️ Solution Overview

### **Approach:**
1. **Two Model Implementations:**
   - **Model A:** Original LeNet-5 with regular convolutions
   - **Model B:** Modified LeNet-5 with depthwise separable convolutions

2. **Training & Evaluation:**
   - Train both models on MNIST for 15 epochs
   - Plot accuracy vs epochs and loss vs epochs
   - Compare final accuracy and convergence behavior

3. **Operation Analysis:**
   - Count multiply-add operations (MACs) for each layer
   - Calculate total operations for entire network
   - Derive mathematical function for operation mapping

4. **Mathematical Derivation:**
   - Prove formula: \( f(x, C_{out}, K) = x \times \left(\frac{1}{C_{out}} + \frac{1}{K^2}\right) \)
   - Verify with actual layer calculations

---

## 📊 Quick Results

| Aspect | Regular LeNet-5 | Depthwise LeNet-5 | Difference |
|--------|----------------|------------------|------------|
| **Final Test Accuracy** | 98.87% | 96.79% | -2.08% |
| **Total Operations** | 804,776 | 94,624 | **-88.24%** |
| **MAC Operations** | 405,600 | 51,224 | **-87.37%** |
| **Multiplications** | 405,600 | 51,224 | **-87.37%** |
| **Additions** | 399,176 | 43,400 | **-89.13%** |
| **Final Loss** | 0.0060 | 0.0684 | +0.0624 |
| **Training Time** | ~2 min | ~2.5 min | +0.5 min |

### **Layer-wise Reduction:**
| Layer | Reduction | Regular Ops | Depthwise Ops |
|-------|-----------|-------------|---------------|
| Conv1 | 79.33% | 117,600 | 24,304 |
| Conv2 | 89.75% | 240,000 | 24,600 |
| Conv3 | 95.17% | 48,000 | 2,320 |

---

## 🏗️ Architecture Comparison

### **Original LeNet-5 Architecture:**



---

## 🧮 Mathematical Derivation

### **Problem:**
Derive \( f(x, \text{parameters}) = y \) where:
- \( x \): Operations in regular convolution
- \( y \): Operations in depthwise separable convolution
- Same output dimensions

### **Solution:**

**Step 1: Define Parameters**
\[
\begin{aligned}
& C_{in}: \text{Input channels} \\
& C_{out}: \text{Output channels} \\
& K: \text{Kernel size} \\
& N = H_{out} \times W_{out}: \text{Output spatial positions}
\end{aligned}
\]

**Step 2: Regular Convolution Operations**
\[
x = C_{out} \times N \times C_{in} \times K^2
\]

**Step 3: Depthwise Separable Operations**
\[
y = \underbrace{C_{in} \times N \times K^2}_{\text{Depthwise}} + \underbrace{C_{out} \times N \times C_{in}}_{\text{Pointwise}}
\]

**Step 4: Express in Terms of \( x \)**
From Step 2:
\[
N \times C_{in} = \frac{x}{C_{out} \times K^2}
\]

Substitute into Step 3:
\[
y = \frac{x}{C_{out} \times K^2} \times (K^2 + C_{out})
\]

**Step 5: Final Function**
\[
\boxed{f(x, C_{out}, K) = x \times \left(\frac{1}{C_{out}} + \frac{1}{K^2}\right)}
\]

### **Verification with Conv1:**
\[
\begin{aligned}
x &= 117,600, \quad C_{out} = 6, \quad K = 5 \\
y &= 117,600 \times \left(\frac{1}{6} + \frac{1}{25}\right) \\
  &= 117,600 \times 0.2067 = 24,304 \quad \text{✓ Matches actual}
\end{aligned}
\]

---

## 🚀 Installation & Setup

### **Prerequisites**
- Python 3.8 or higher
- pip package manager
- 2GB+ free disk space for MNIST dataset

### **Method 1: Using requirements.txt**
```bash
# Clone repository
git clone https://github.com/yourusername/lenet5-depthwise-convolution.git
cd lenet5-depthwise-convolution

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
