import os 
import sys 
current_dir = os.getcwd()  # Get current working directory
parent_dir = os.path.dirname(current_dir)  # Get parent directory
sys.path.append(parent_dir)
from typing import Tuple
import numpy as np


from .tree import TreeNode, BinaryTree

########################################
# Label Tree
########################################

class LabelNode(TreeNode):
    data_type = list
    def __init__(self, val=None, left=None, right=None):
        """Binary tree node class where each node contains a set of labels
        
        Args:
            val: Node value/data (must be a list of labels)
            left: Left child node
            right: Right child node
        """
        assert val is None or isinstance(val, list), "Node value must be a list of labels"
        super().__init__(val, left, right)  # Initialize using parent class constructor
    
    def __str__(self):
        return f"{self.val}"
    
    def is_leaf(self):
        """Check if node is a leaf node"""
        return self.left is None and self.right is None

class LabelBinaryTree(BinaryTree):
    def __init__(self):
        """Initialize empty binary tree that inherits from BinaryTree"""
        super().__init__()  # Initialize parent class
        self.node_type = LabelNode
        
        

    
  