#include <vector>

class Solution {
public:
    int search(std::vector<int>& nums, int target) {
        int left = 0;
        int right = nums.size() - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            // Target found
            if (nums[mid] == target) {
                return mid;
            }
            
            // Check if the LEFT half is perfectly sorted
            if (nums[left] <= nums[mid]) {
                // Check if the target falls within this sorted left half's range
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1; // Target is in the left half
                } else {
                    left = mid + 1;  // Target must be in the right half
                }
            } 
           
            else {
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;  // Target is in the right half
                } else {
                    right = mid - 1; // Target must be in the left half
                }
            }
        }
        
        return -1; // Target not found
    }
};