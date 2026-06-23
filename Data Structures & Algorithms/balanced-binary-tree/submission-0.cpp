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
#include <algorithm>
#include <cmath>

class Solution {
public:
    bool isBalanced(TreeNode* root) {
        // If the helper function returns -1, the tree is unbalanced.
        // Otherwise, it returns the actual height, meaning it is balanced.
        return checkHeight(root) != -1;
    }

private:
    int checkHeight(TreeNode* node) {
        // Base case: An empty tree has a height of 0
        if (node == nullptr) {
            return 0;
        }
        
        // Check the left subtree. If it's unbalanced, short-circuit and propagate -1.
        int leftHeight = checkHeight(node->left);
        if (leftHeight == -1) return -1;
        
        // Check the right subtree. If it's unbalanced, short-circuit and propagate -1.
        int rightHeight = checkHeight(node->right);
        if (rightHeight == -1) return -1;
        
        // Check if the current node is unbalanced
        if (std::abs(leftHeight - rightHeight) > 1) {
            return -1; // Imbalance found!
        }
        
        // If it is balanced, return its actual height to its parent
        return 1 + std::max(leftHeight, rightHeight);
    }
};