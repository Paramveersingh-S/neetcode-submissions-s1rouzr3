# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root: TreeNode) -> str:
        """Encodes a tree to a single string."""
        res = []
        
        def dfs(node):
            # Base case: if node is null, append our marker
            if not node:
                res.append("N")
                return
            
            # Preorder: Root -> Left -> Right
            res.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
            
        dfs(root)
        
        # Join the list into a single string separated by commas
        return ",".join(res)
        

    def deserialize(self, data: str) -> TreeNode:
        """Decodes your encoded data to tree."""
        # Split the string back into a list of values
        vals = data.split(",")
        
        # We use an iterator so we can consume values one by one globally
        # using the next() function.
        self.i = 0 
        
        def dfs():
            # Get the current value and move our pointer forward
            val = vals[self.i]
            self.i += 1
            
            # If it's a null marker, this branch is a dead end
            if val == "N":
                return None
            
            # Create the current node
            node = TreeNode(int(val))
            
            # Preorder reconstruction: build left child first, then right child
            node.left = dfs()
            node.right = dfs()
            
            return node
            
        return dfs()

# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))