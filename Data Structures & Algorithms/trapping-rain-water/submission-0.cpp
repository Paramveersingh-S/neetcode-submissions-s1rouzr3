#include <vector>

class Solution {
public:
    int trap(std::vector<int>& height) {
        if (height.empty()) return 0;
        
        int left = 0;
        int right = height.size() - 1;
        
        int left_max = height[left];
        int right_max = height[right];
        
        int total_water = 0;
        
        while (left < right) {
            if (left_max < right_max) {
                left++;
                if (height[left] >= left_max) {
                    left_max = height[left];
                } else {
                    total_water += left_max - height[left];
                }
            } else {
                right--;
                if (height[right] >= right_max) {
                    right_max = height[right];
                } else {
                    total_water += right_max - height[right];
                }
            }
        }
        
        return total_water;
    }
};