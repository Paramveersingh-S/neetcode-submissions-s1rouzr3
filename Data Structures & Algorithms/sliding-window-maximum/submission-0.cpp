#include <vector>
#include <deque>

class Solution {
public:
    std::vector<int> maxSlidingWindow(std::vector<int>& nums, int k) {
        std::vector<int> result;
        std::deque<int> dq;
        
        for (int i = 0; i < nums.size(); ++i) {
            // Remove indices that are out of the current window bounds
            if (!dq.empty() && dq.front() < i - k + 1) {
                dq.pop_front();
            }
            
            // Remove indices of smaller elements as they are useless now
            while (!dq.empty() && nums[dq.back()] <= nums[i]) {
                dq.pop_back();
            }
            
            // Add the current element's index
            dq.push_back(i);
            
            // Once our window reaches size k, the front of the deque is our maximum
            if (i >= k - 1) {
                result.push_back(nums[dq.front()]);
            }
        }
        
        return result;
    }
};
