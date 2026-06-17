#include <vector>
#include <algorithm>
#include <climits>

class Solution {
public:
    double findMedianSortedArrays(std::vector<int>& nums1, std::vector<int>& nums2) {
        // Ensure we always binary search on the smaller array for O(log(min(m,n)))
        if (nums1.size() > nums2.size()) {
            return findMedianSortedArrays(nums2, nums1);
        }
        
        int m = nums1.size();
        int n = nums2.size();
        int left = 0;
        int right = m;
        
        // This formula handles both even and odd total lengths elegantly
        int half_len = (m + n + 1) / 2; 
        
        while (left <= right) {
            int cut1 = left + (right - left) / 2;
            int cut2 = half_len - cut1;
            
            // Handle edge cases where the cut is at the absolute extreme edges
            int maxLeft1 = (cut1 == 0) ? INT_MIN : nums1[cut1 - 1];
            int minRight1 = (cut1 == m) ? INT_MAX : nums1[cut1];
            
            int maxLeft2 = (cut2 == 0) ? INT_MIN : nums2[cut2 - 1];
            int minRight2 = (cut2 == n) ? INT_MAX : nums2[cut2];
            
            // Check if we found the perfect valid partition
            if (maxLeft1 <= minRight2 && maxLeft2 <= minRight1) {
                // If total length is even
                if ((m + n) % 2 == 0) {
                    return (std::max(maxLeft1, maxLeft2) + std::min(minRight1, minRight2)) / 2.0;
                } 
                // If total length is odd
                else {
                    return std::max(maxLeft1, maxLeft2);
                }
            }
            // If maxLeft1 is too large, we must move cut1 to the left
            else if (maxLeft1 > minRight2) {
                right = cut1 - 1;
            }
            // If maxLeft2 is too large, we must move cut1 to the right
            else {
                left = cut1 + 1;
            }
        }
        
        return 0.0; // Fallback, should theoretically never be reached if arrays are sorted
    }
};