/**
 * Definition for a binary tree node.
 * struct TreeNode {
 * int val;
 * TreeNode *left;
 * TreeNode *right;
 * TreeNode() : val(0), left(nullptr), right(nullptr) {}
 * TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 * TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int diameterOfBinaryTree(TreeNode* root) {
        int max_diameter = 0;
        
        // Kick off the DFS traversal
        calculateDepth(root, max_diameter);
        
        return max_diameter;
    }

private:
    // Helper function that returns the depth of a subtree 
    // while simultaneously updating the global max diameter.
    int calculateDepth(TreeNode* node, int& max_diameter) {
        if (node == nullptr) {
            return 0; // An empty node has 0 depth
        }
        
        // Recursively find the depth of the left and right subtrees
        int left_depth = calculateDepth(node->left, max_diameter);
        int right_depth = calculateDepth(node->right, max_diameter);
        
        // The longest path passing through the CURRENT node is the sum of its subtrees' depths
        max_diameter = std::max(max_diameter, left_depth + right_depth);
        
        // Return the actual depth of this node so its parent can use it
        return 1 + std::max(left_depth, right_depth);
    }
};