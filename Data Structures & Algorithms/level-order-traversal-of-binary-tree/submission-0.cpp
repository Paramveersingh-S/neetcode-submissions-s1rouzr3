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
#include <vector>
#include <queue>

class Solution {
public:
    std::vector<std::vector<int>> levelOrder(TreeNode* root) {
        std::vector<std::vector<int>> result;
        
        // Edge case: If the tree is completely empty, return the empty result
        if (root == nullptr) {
            return result;
        }
        
        std::queue<TreeNode*> q;
        q.push(root);
        
        // Continue traversing as long as there are nodes in the queue
        while (!q.empty()) {
            int level_size = q.size(); // Snapshot the number of nodes at this specific level
            std::vector<int> current_level;
            
            // Process ONLY the nodes that belong to the current level
            for (int i = 0; i < level_size; ++i) {
                TreeNode* node = q.front();
                q.pop();
                
                current_level.push_back(node->val);
                
                // Queue up the next level's nodes
                if (node->left != nullptr) {
                    q.push(node->left);
                }
                if (node->right != nullptr) {
                    q.push(node->right);
                }
            }
            
            // Append the fully processed level to our final result
            result.push_back(current_level);
        }
        
        return result;
    }
};