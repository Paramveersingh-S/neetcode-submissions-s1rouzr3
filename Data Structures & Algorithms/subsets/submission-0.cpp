#include <vector>

class Solution {
public:
    std::vector<std::vector<int>> subsets(std::vector<int>& nums) {
        std::vector<std::vector<int>> result;
        std::vector<int> current;
        
        // Start the backtracking process from index 0
        backtrack(0, nums, current, result);
        
        return result;
    }

private:
    void backtrack(int index, std::vector<int>& nums, std::vector<int>& current, std::vector<std::vector<int>>& result) {
        // Base Case: We have made a decision (Include/Exclude) for every element
        if (index == nums.size()) {
            result.push_back(current);
            return;
        }

        // Decision 1: INCLUDE the current element
        current.push_back(nums[index]);
        backtrack(index + 1, nums, current, result);

        // Decision 2: EXCLUDE the current element
        // We pop the element we just added to revert the state back to what it was
        current.pop_back();
        backtrack(index + 1, nums, current, result);
    }
};