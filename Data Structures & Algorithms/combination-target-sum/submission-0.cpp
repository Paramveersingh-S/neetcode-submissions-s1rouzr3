#include <vector>

class Solution {
public:
    std::vector<std::vector<int>> combinationSum(std::vector<int>& candidates, int target) {
        std::vector<std::vector<int>> result;
        std::vector<int> current;
        
        // Start the recursive backtracking from index 0
        backtrack(0, target, candidates, current, result);
        
        return result;
    }

private:
    void backtrack(int index, int remaining_target, std::vector<int>& candidates, 
                   std::vector<int>& current, std::vector<std::vector<int>>& result) {
        
        // Base Case 1: We hit exactly 0! We found a valid combination.
        if (remaining_target == 0) {
            result.push_back(current);
            return;
        }
        
        // Base Case 2: We overshot the target (negative) OR we ran out of numbers.
        if (remaining_target < 0 || index >= candidates.size()) {
            return;
        }
        
        // Decision 1: INCLUDE the current number. 
        // Notice we do NOT increment the index because we can reuse the same number.
        current.push_back(candidates[index]);
        backtrack(index, remaining_target - candidates[index], candidates, current, result);
        
        // Decision 2: EXCLUDE the current number and MOVE to the next index.
        // We backtrack by popping the number we just added, then increment the index.
        current.pop_back();
        backtrack(index + 1, remaining_target, candidates, current, result);
    }
};