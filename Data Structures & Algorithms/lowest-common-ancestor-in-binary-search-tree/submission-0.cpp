/**
 * Definition for a binary tree node.
 * struct TreeNode {
 * int val;
 * TreeNode *left;
 * TreeNode *right;
 * TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */

class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        // We use an iterative approach to achieve O(1) auxiliary space.
        // We just traverse down the tree, replacing 'root' with the next step.
        while (root != nullptr) {
            
            // If both p and q are strictly less than the current node,
            // the LCA must be entirely in the left subtree.
            if (p->val < root->val && q->val < root->val) {
                root = root->left;
            } 
            // If both p and q are strictly greater than the current node,
            // the LCA must be entirely in the right subtree.
            else if (p->val > root->val && q->val > root->val) {
                root = root->right;
            } 
            // If they split (one is smaller, one is larger), OR if one of them 
            // equals the current node, we have found our divergence point.
            else {
                return root;
            }
        }
        
        return nullptr; // Fallback (shouldn't be reached if p and q exist in the BST)
    }
};