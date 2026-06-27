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

class Solution {
public:
    int goodNodes(TreeNode* root) {
        // Kick off the DFS using the root's value as the initial maximum
        return dfs(root, root->val);
    }
    
private:
    int dfs(TreeNode* node, int max_so_far) {
        // Base Case: We've hit an empty leaf path
        if (node == nullptr) {
            return 0;
        }
        
        int count = 0;
        
        // If the current node is greater than or equal to the max seen on 
        // this path so far, it qualifies as a "good" node!
        if (node->val >= max_so_far) {
            count = 1;
        }
        
        // Update our "backpack" to carry the largest value seen so far
        int new_max = std::max(max_so_far, node->val);
        
        // Ask the left and right subtrees how many good nodes they found,
        // passing down the updated maximum value constraint.
        count += dfs(node->left, new_max);
        count += dfs(node->right, new_max);
        
        return count;
    }
};