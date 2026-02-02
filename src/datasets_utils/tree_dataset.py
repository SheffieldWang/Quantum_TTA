
import numpy as np
from mindspore.dataset import NumpySlicesDataset
from .dataset import get_dataloader_with_idx_list








def label_transform(label,label_binary_tree,class_idx_list):
    assert len(class_idx_list) >1 , "class_idx_list must be a list with more than 1 element"
    class_idx_list_node = label_binary_tree.find_node(class_idx_list)
    left_class_idx_list = class_idx_list_node.left.val
    right_class_idx_list = class_idx_list_node.right.val
    
    # Handle both single labels and arrays of labels
    if hasattr(label, '__len__') and not isinstance(label, str):
        # It's an array-like object

        result = np.zeros_like(label)
        left_mask = np.isin(label, left_class_idx_list)
        right_mask = np.isin(label, right_class_idx_list)
        
        result[left_mask] = -1
        result[right_mask] = 1
        
        # Check if any labels are not found
        valid_mask = left_mask | right_mask
        if not np.all(valid_mask):
            invalid_labels = label[~valid_mask]
            raise ValueError(f"Labels {invalid_labels} not found in class_idx_list {class_idx_list}")
        
        return result


    
    
def get_tree_dataloaders(n_qubits, n_layers, n_train_samples, n_test_samples, batch_size,data_type=None,label_binary_tree=None,class_idx_list=None):
    """
    Get the DataLoader for the tree-based classification dataset
    
    Args:
        n_qubits (int): Number of qubits
        n_layers (int): Number of layers
        batch_size (int): Batch size, defaults to 200
        data_type (str): Type of dataset to load
        labeltree (BinaryTree): Binary tree structure for label classification
        class_idx_list (list): List of class indices for the current tree node
    
    Returns:
        tuple: (train_loader, test_loader) - DataLoaders for training and testing
    
    Usage:
        This function creates DataLoaders that transform labels according to the binary tree structure.
        Labels are classified as -1 (left subtree) or 1 (right subtree) based on the tree node.
    """
    
    train_loader,test_loader = get_dataloader_with_idx_list(n_qubits=n_qubits, n_layers=n_layers, n_train_samples=n_train_samples, n_test_samples=n_test_samples, batch_size=batch_size, data_type=data_type, class_idx_list=class_idx_list)
    
    
    train_data = []
    train_labels = []

    for data, label in train_loader:
        train_data.append(data)
        train_labels.append(label)

    # Convert to numpy arrays if needed
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    train_labels_tree = label_transform(train_labels,label_binary_tree,class_idx_list)

    test_data = []
    test_labels = []

    for data, label in test_loader:
        test_data.append(data)
        test_labels.append(label)

    test_data = np.concatenate(test_data, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_labels_tree = label_transform(test_labels,label_binary_tree,class_idx_list)
    
    train_loader = NumpySlicesDataset({"data": train_data, "labels": train_labels_tree},shuffle=False).batch(batch_size)
    test_loader = NumpySlicesDataset({"data": test_data, "labels": test_labels_tree},shuffle=False).batch(batch_size)

    return train_loader,test_loader
