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
    bool isSameTree(TreeNode* p, TreeNode* q) {
        // Base Case 1: Both nodes are null. The structure matches here.
        if (p == nullptr && q == nullptr) {
            return true;
        }
        
        // Base Case 2: One is null but the other is not. The structure is different.
        // (We already know they aren't BOTH null because of the first check)
        if (p == nullptr || q == nullptr) {
            return false;
        }
        
        // Base Case 3: Both nodes exist, but their values are different.
        if (p->val != q->val) {
            return false;
        }
        
        // If we survived all base cases, the current nodes are identical.
        // Now, strictly verify that BOTH the left and right subtrees are also identical.
        return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
    }
};