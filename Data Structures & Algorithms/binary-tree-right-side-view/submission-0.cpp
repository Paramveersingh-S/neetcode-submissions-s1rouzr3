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
    std::vector<int> rightSideView(TreeNode* root) {
        std::vector<int> result;
        
        // Edge case: Empty tree
        if (root == nullptr) {
            return result;
        }
        
        std::queue<TreeNode*> q;
        q.push(root);
        
        // Traverse level by level using BFS
        while (!q.empty()) {
            int level_size = q.size();
            
            // Iterate through all nodes on the current level
            for (int i = 0; i < level_size; ++i) {
                TreeNode* node = q.front();
                q.pop();
                
                // If this is the last node in the current level's loop,
                // it is the rightmost node. Add it to the result!
                if (i == level_size - 1) {
                    result.push_back(node->val);
                }
                
                // Push children into the queue for the next level
                // (Always push left first, then right, so the rightmost ends up at the back)
                if (node->left != nullptr) {
                    q.push(node->left);
                }
                if (node->right != nullptr) {
                    q.push(node->right);
                }
            }
        }
        
        return result;
    }
};