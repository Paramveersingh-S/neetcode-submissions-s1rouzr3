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
    bool isValidBST(TreeNode* root) {
        // Kick off the DFS with no initial boundaries (represented by nullptrs)
        return validate(root, nullptr, nullptr);
    }

private:
    bool validate(TreeNode* node, TreeNode* minNode, TreeNode* maxNode) {
        // Base Case: An empty node is technically a valid BST
        if (node == nullptr) {
            return true;
        }
        
        // Check the lower boundary constraint
        // If minNode is set, the current node MUST be strictly greater than minNode
        if (minNode != nullptr && node->val <= minNode->val) {
            return false;
        }
        
        // Check the upper boundary constraint
        // If maxNode is set, the current node MUST be strictly less than maxNode
        if (maxNode != nullptr && node->val >= maxNode->val) {
            return false;
        }
        
        // Recursively validate the left and right subtrees.
        // When going LEFT, the current node becomes the new STRICT MAXIMUM.
        // When going RIGHT, the current node becomes the new STRICT MINIMUM.
        return validate(node->left, minNode, node) && 
               validate(node->right, node, maxNode);
    }
};