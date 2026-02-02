# QAdaBoost: Quantum AdaBoost for Multi-class Classification

## Project Overview

QAdaBoost is a multi-class classification framework based on quantum computing and the AdaBoost algorithm. This project combines quantum neural networks (QNN) with the classical AdaBoost algorithm, implementing efficient multi-class classification tasks through binary tree structures. The project uses the MindSpore Quantum framework for quantum circuit implementation and employs hierarchical classification strategies to enhance classification performance.

## Key Features

- **Quantum AdaBoost Algorithm**: Ensemble learning method combining quantum neural networks and AdaBoost
- **Binary Tree Multi-class Classification**: Decomposes multi-class problems into multiple binary classification problems through binary tree structures
- **MindSpore Quantum**: Quantum circuit implementation based on Huawei's MindSpore Quantum framework
- **Scalable Architecture**: Modular design supporting different datasets and quantum circuit configurations
- **High Performance**: Achieves high accuracy on MNIST and custom classification datasets

## Project Structure

```
QAdaboost_Huawei/
├── configs/                    # Configuration files
│   └── tree_classification_012.json
├── datasets/                   # Preprocessed datasets
│   ├── classification_012/    # 3-class classification dataset
│   ├── mnist/                 # Full MNIST dataset
│   └── mnist_036147/          # MNIST subset (digits 0,3,6,1,4,7)
├── notebooks/                  # Jupyter Notebook examples
│   ├── datasets_generation/   # Dataset generation
│   ├── example/               # Training examples
│   ├── label_tree_generation/ # Label tree generation
│   └── trace_distance_test/   # Trace distance testing
├── raw_datasets/              # Raw datasets
│   └── MNIST/
├── results/                   # Experimental results
│   └── tree_results/          # Label tree results
└── src/                       # Source code
    ├── adaboost/              # AdaBoost implementation
    ├── datasets_utils/        # Dataset utilities
    ├── model/                 # Quantum models
    ├── train_utils/           # Training utilities
    └── tree_utils/            # Tree structure utilities
```

## Core Algorithms

### 1. Quantum AdaBoost Algorithm

The QAdaBoost algorithm combines the classical AdaBoost algorithm with quantum neural networks:

```python
class QAdaBoost:
    def __init__(self, n_train):
        self.n_estimators = 0
        self.models = []      # Quantum base classifiers
        self.alphas = []      # Classifier weights
        self.errors = []      # Classification error rates
        self.weights = np.ones(n_train) / n_train  # Sample weights
```

### 2. Binary Tree Multi-class Strategy

Decomposes multi-class problems into binary tree structures:
- Root node: All classes
- Internal nodes: Subsets of classes
- Leaf nodes: Individual classes

Example tree structure:
```
  _[0, 1, 2]______      
 /                \     
[2]           _[0, 1]_  
             /        \ 
            [0]      [1]
```

### 3. Quantum Circuit Design

Uses RY encoder and parameterized quantum circuits:

```python
class QuantumCircuit:
    def __init__(self, n_qubits, n_layers):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.simulator = Simulator('mqvector', n_qubits)
        self.observable = Hamiltonian(QubitOperator('Z0', 1))
```

## Installation Requirements

### System Requirements
- Python 3.8+
- Linux/Windows/macOS

### Dependencies
```bash
# Core dependencies
mindspore>=2.0.0
mindquantum>=0.9.0
numpy>=1.21.0
matplotlib>=3.5.0

# Optional dependencies
wandb>=0.15.0  # Experiment tracking
tqdm>=4.64.0   # Progress bars
```

### Installation Steps
1. Clone the project:
```bash
git clone https://github.com/SheffieldWang/Quantum-Trace-distance-binary-Tree-AdaBoost-classifier.git
cd Quantum-Trace-distance-binary-Tree-AdaBoost-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import mindspore; import mindquantum; print('Installation successful')"
```

## Training Example

Run the example training script:

```python
# Using Jupyter Notebook
jupyter notebook notebooks/example/train_classification_012.ipynb
```

