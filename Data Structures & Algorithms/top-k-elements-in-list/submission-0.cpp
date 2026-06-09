#include <vector>
#include <unordered_map>

class Solution {
public:
    std::vector<int> topKFrequent(std::vector<int>& nums, int k) {
        std::unordered_map<int, int> count;
        for (int num : nums) {
            count[num]++;
        }
        
        std::vector<std::vector<int>> buckets(nums.size() + 1);
        for (auto& pair : count) {
            buckets[pair.second].push_back(pair.first);
        }
        
        std::vector<int> result;
        for (int i = buckets.size() - 1; i >= 0; --i) {
            for (int num : buckets[i]) {
                result.push_back(num);
                if (result.size() == k) {
                    return result;
                }
            }
        }
        
        return result;
    }
};