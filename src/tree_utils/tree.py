## Basic Binary Tree 

class TreeNode:
    data_type = None
    def __init__(self, val=0, left=None, right=None):
        """Basic binary tree node class
        
        Args:
            val: Node value/data
            left: Left child node
            right: Right child node
        """
        
        self.val = val
        self.left = left
        self.right = right
        
        
    def __str__(self):
        raise NotImplementedError("Subclasses must implement __str__ method")
        
        
class BinaryTree:
    def __init__(self, root=None):
        """Basic binary tree implementation
        
        Args:
            root: Root node of the tree (default None for empty tree)
        """
        self.root = root
        self.node_type = TreeNode  # Store the node type this tree accepts
        self.data_type = self.node_type.data_type

    def insert(self, val, position='left', parent_val=None):
        """Insert new value into binary tree with position control
        Args:
            val: Value to insert (must be a list)
            position: 'left' or 'right' to specify insertion side
            parent_val: Value of parent node to insert under (None for root)
        """
        if self.data_type is None:
            if val is None:
                raise ValueError("Must specify TreeNode data type - cannot insert None value")            
        else:
            assert isinstance(val, self.data_type), f"Can only insert {self.data_type} into this tree"
            assert position in ['left', 'right'], "Position must be 'left' or 'right'"
        
        if self.root is None:
            self.root = self.node_type(val)
        else:
            if parent_val is None:
                # Insert at root level
                self._insert_recursive(self.root, val, position)
            else:
                # Find parent node and insert under it
                parent_node = self._find_node(self.root, parent_val)
                if parent_node:
                    self._insert_recursive(parent_node, val, position)
                else:
                    raise ValueError("Parent node not found")

    def find_node(self, target_val):
        """Find node with given value in the tree
        
        Args:
            target_val: Value to search for
            
        Returns:
            TreeNode: Node with matching value, or None if not found
        """
        return self._find_node(self.root, target_val)

    def _find_node(self, node, target_val):
        """Helper to find node with given value"""
        if node is None:
            return None
        if node.val == target_val:
            return node
        left_result = self._find_node(node.left, target_val)
        if left_result:
            return left_result
        return self._find_node(node.right, target_val)
    

    def _insert_recursive(self, node, val, position):
        """Recursive insert helper method with position control"""
        if position == 'left':
            if node.left is None:
                node.left = self.node_type(val)
            else:
                self._insert_recursive(node.left, val, position)
        else:
            if node.right is None:
                node.right = self.node_type(val)
            else:
                self._insert_recursive(node.right, val, position)

    def traverse(self, node=None):
        """In-order traversal of binary tree
        Args:
            node: Starting node (defaults to root)
        Returns:
            List of values in in-order traversal sequence
        """
        if node is None:
            node = self.root
            if node is None:
                return []
        
        result = []
        if node.left:
            result += self.traverse(node.left)
        result.append(node.val)
        if node.right:
            result += self.traverse(node.right)
        return result

    def get_non_leaf_nodes(self):
        """获取二叉树中所有非叶子节点
        
        Returns:
            List: 包含所有非叶子节点值的列表
        """
        non_leaf_nodes = []
        
        def traverse(node):
            if node is None:
                return
            
            # 如果节点有左子节点或右子节点，则它是非叶子节点
            if node.left is not None or node.right is not None:
                non_leaf_nodes.append(node.val)
            
            # 递归遍历左右子树
            traverse(node.left)
            traverse(node.right)
        
        # 从根节点开始遍历
        traverse(self.root)
        return non_leaf_nodes

    def get_leaf_nodes(self):
        """获取二叉树中所有叶子节点
        
        Returns:
            List: 包含所有叶子节点值的列表
        """
        leaf_nodes = []
        
        def traverse(node):
            if node is None:
                return
            
            # 如果节点没有左子节点和右子节点，则它是叶子节点
            if node.left is None and node.right is None:
                leaf_nodes.append(node.val)
            
            # 递归遍历左右子树
            traverse(node.left)
            traverse(node.right)
        
        # 从根节点开始遍历
        traverse(self.root)
        return leaf_nodes

    def visualize(self):
        """Visualize the binary tree using ASCII art"""
        if self.root is None:
            print("Empty tree")
            return
        
        lines, *_ = self._visualize_helper(self.root)
        for line in lines:
            print(line)

    def _visualize_helper(self, node):
        """Returns list of strings, width, height, and horizontal coordinate of the root"""
        if node.right is None and node.left is None:
            line = str(node)
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child
        if node.right is None:
            lines, n, p, x = self._visualize_helper(node.left)
            s = str(node)
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child
        if node.left is None:
            lines, n, p, x = self._visualize_helper(node.right)
            s = str(node)
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children
        left, n, p, x = self._visualize_helper(node.left)
        right, m, q, y = self._visualize_helper(node.right)
        s = str(node)
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2
