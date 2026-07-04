# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def buildTree(self, preorder: list[int], inorder: list[int]) -> Optional[TreeNode]:
        # Hash map to instantly find the index of a value in the inorder array
        # Format: {node_value: index_in_inorder}
        inorder_map = {val: i for i, val in enumerate(inorder)}
        
        # Pointer to track which node in preorder we are currently turning into a root
        preorder_index = 0
        
        # Recursive helper function using left and right boundaries of the inorder array
        def build(left: int, right: int) -> Optional[TreeNode]:
            # nonlocal allows us to modify the preorder_index variable defined outside this function
            nonlocal preorder_index
            
            # Base case: if left boundary passes right boundary, there are no nodes to build
            if left > right:
                return None
                
            # 1. Get the current root value from preorder and create the node
            root_val = preorder[preorder_index]
            root = TreeNode(root_val)
            preorder_index += 1
            
            # 2. Find where this root splits the inorder array
            mid = inorder_map[root_val]
            
            # 3. Recursively build the left and right subtrees
            # Everything left of 'mid' forms the left subtree
            root.left = build(left, mid - 1)
            
            # Everything right of 'mid' forms the right subtree
            root.right = build(mid + 1, right)
            
            return root
            
        # Start the recursion with the full bounds of the inorder array
        return build(0, len(inorder) - 1)