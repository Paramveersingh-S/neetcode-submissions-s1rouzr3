class Solution:
    def combinationSum2(self, candidates: list[int], target: int) -> list[list[int]]:
        candidates.sort()
        res = []
        
        def backtrack(start_index, current_combo, remaining_target):
            if remaining_target == 0:
                res.append(list(current_combo))
                return
            
            for i in range(start_index, len(candidates)):
                if i > start_index and candidates[i] == candidates[i - 1]:
                    continue
                    
                if candidates[i] > remaining_target:
                    break
                current_combo.append(candidates[i])
                backtrack(i + 1, current_combo, remaining_target - candidates[i])
                current_combo.pop()
        backtrack(0, [], target)
        
        return res