# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        # Initialize to negative infinity so even negative node values can update it
        global_max = float('-inf')

        # Helper function for our Post-order DFS
        def dfs(node):
            nonlocal global_max
            
            # Base case: an empty node contributes 0 to a path sum
            if not node:
                return 0

            # 1. Evaluate children first (Post-order)
            # If a branch is negative, we use max(0, ...) to effectively ignore it
            left_branch_max = max(0, dfs(node.left))
            right_branch_max = max(0, dfs(node.right))

            # 2. Peak Check: What if THIS node is the highest point of the path?
            current_peak_sum = node.val + left_branch_max + right_branch_max
            
            # Update the global maximum if this peak is the best we've seen
            global_max = max(global_max, current_peak_sum)

            # 3. Return to parent: We can only continue the path down ONE branch
            return node.val + max(left_branch_max, right_branch_max)

        # Start the traversal from the root
        dfs(root)
        
        # After the entire tree is evaluated, our global_max holds the answer
        return global_max