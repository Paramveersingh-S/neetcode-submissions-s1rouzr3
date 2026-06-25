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
    bool isSubtree(TreeNode* root, TreeNode* subRoot) {
        // Base Case 1: An empty subRoot is technically a subtree of any tree
        if (subRoot == nullptr) return true;
        
        // Base Case 2: If the main tree is empty but subRoot is not, we can't find it
        if (root == nullptr) return false;
        
        // Check if the trees match perfectly starting from the current node
        if (isSameTree(root, subRoot)) {
            return true;
        }
        
        // If not, recursively search down both the left and right branches of the main tree.
        // We use OR (||) because we only need to find the subtree in ONE of the branches.
        return isSubtree(root->left, subRoot) || isSubtree(root->right, subRoot);
    }

private:
    // Our trusted helper function from the 'Same Binary Tree' problem
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if (p == nullptr && q == nullptr) return true;
        if (p == nullptr || q == nullptr) return false;
        if (p->val != q->val) return false;
        
        return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
    }
};