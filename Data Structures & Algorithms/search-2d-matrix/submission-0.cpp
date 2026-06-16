#include <vector>

class Solution {
public:
    bool searchMatrix(std::vector<std::vector<int>>& matrix, int target) {
        if (matrix.empty() || matrix[0].empty()) {
            return false;
        }
        
        int m = matrix.size();
        int n = matrix[0].size();
        
        int left = 0;
        int right = m * n - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            // Map the 1D index to a 2D row and column
            int row = mid / n;
            int col = mid % n;
            
            int mid_value = matrix[row][col];
            
            if (mid_value == target) {
                return true;
            } else if (mid_value < target) {
                left = mid + 1; // Target is in the right half
            } else {
                right = mid - 1; // Target is in the left half
            }
        }
        
        return false;
    }
};