# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        stack = []
        curr = root
        
        # We loop as long as there are nodes to process in the stack, 
        # or we have a current node we are actively looking at.
        while stack or curr:
            
            # Keep diving left until we hit a dead end (null)
            # This ensures we always reach the absolute smallest remaining number
            while curr:
                stack.append(curr)
                curr = curr.left
                
            # Pop the top node from the stack (the next smallest number)
            curr = stack.pop()
            
            # We "visited" a node in sorted order, so we decrement our counter
            k -= 1
            
            # If k is 0, we've found our kth smallest number
            if k == 0:
                return curr.val
                
            # After processing the node, we must check if it has any numbers
            # on its right side (which are larger than it, but smaller than its parent)
            curr = curr.right
            
        return -1 # Fallback, though the problem guarantees k is valid