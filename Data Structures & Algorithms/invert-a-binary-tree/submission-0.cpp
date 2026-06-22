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
    TreeNode* invertTree(TreeNode* root) {
        // Base Case: If the tree (or subtree) is empty, do nothing.
        if (root == nullptr) {
            return nullptr;
        }
        
        // 1. Store the left child in a temporary variable
        TreeNode* temp = root->left;
        
        // 2. Swap the left and right children
        root->left = root->right;
        root->right = temp;
        
        // 3. Recursively ask the left and right subtrees to invert themselves
        invertTree(root->left);
        invertTree(root->right);
        
        // Return the root of the newly inverted tree
        return root;
    }
};
