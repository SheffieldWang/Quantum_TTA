import os
import numpy as np
from pathlib import Path
from mindspore.dataset import NumpySlicesDataset



def get_raw_dataset(n_qubits, n_layers, n_train_samples, n_test_samples,data_type=None):
    if data_type is None:
        raise ValueError("data_type must be specified.")
    current_dir = Path(__file__).resolve().parent
    dataset_dir = current_dir.parent.parent / "datasets"
    train_data_path = os.path.join(dataset_dir, data_type, f"x_train_qubit_{n_qubits}_layer_{n_layers}_samples_{n_train_samples}.npy")
    train_labels_path = os.path.join(dataset_dir, data_type, f"y_train_qubit_{n_qubits}_layer_{n_layers}_samples_{n_train_samples}.npy")
    test_data_path = os.path.join(dataset_dir, data_type, f"x_test_qubit_{n_qubits}_layer_{n_layers}_samples_{n_test_samples}.npy")
    test_labels_path = os.path.join(dataset_dir, data_type, f"y_test_qubit_{n_qubits}_layer_{n_layers}_samples_{n_test_samples}.npy")
    train_data = np.load(train_data_path)
    train_labels = np.load(train_labels_path)
    test_data = np.load(test_data_path)
    test_labels = np.load(test_labels_path)
    return train_data, train_labels, test_data, test_labels

def get_dataloaders(n_qubits, n_layers, n_train_samples, n_test_samples,batch_size,data_type=None):
    """
    Get the DataLoader for the quantum dataset
    
    Args:
        n_qubits (int): Number of qubits
        n_layers (int): Number of layers
        batch_size (int): Batch size, defaults to 32
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    train_data, train_labels, test_data, test_labels = get_raw_dataset(n_qubits, n_layers, n_train_samples, n_test_samples, data_type)
    
    train_loader = NumpySlicesDataset({"data": train_data, "labels": train_labels},shuffle=False).batch(batch_size)
    test_loader = NumpySlicesDataset({"data": test_data, "labels": test_labels},shuffle=False).batch(batch_size)

    return train_loader, test_loader

def get_dataloader_with_idx(n_qubits, n_layers, n_train_samples, n_test_samples, batch_size, data_type=None, class_idx=None):

    if class_idx is None:
        raise ValueError("class_idx must be specified.")

    train_data, train_labels, test_data, test_labels = get_raw_dataset(n_qubits, n_layers, n_train_samples, n_test_samples, data_type)

    train_mask = train_labels == class_idx
    test_mask = test_labels == class_idx

    train_loader = NumpySlicesDataset({"data": train_data[train_mask], "labels": train_labels[train_mask]},shuffle=False).batch(batch_size)
    test_loader = NumpySlicesDataset({"data": test_data[test_mask], "labels": test_labels[test_mask]},shuffle=False).batch(batch_size)
    
    return train_loader, test_loader


def get_dataloader_with_idx_list(n_qubits, n_layers, n_train_samples, n_test_samples, batch_size, data_type=None, class_idx_list=None):

    if class_idx_list is None:
        raise ValueError("class_idx_list must be specified as List such that [label1, label2, ...]")
   
    x_train, y_train, x_test, y_test = get_raw_dataset(n_qubits, n_layers, n_train_samples, n_test_samples, data_type)
    # Create new training dataset with targets in idx_list
    train_mask = np.isin(y_train, class_idx_list)
    test_mask = np.isin(y_test, class_idx_list)
    
    train_loader = NumpySlicesDataset({"data": x_train[train_mask], "labels": y_train[train_mask]},shuffle=False).batch(batch_size)
    test_loader = NumpySlicesDataset({"data": x_test[test_mask], "labels": y_test[test_mask]},shuffle=False).batch(batch_size)

    
    return train_loader, test_loader