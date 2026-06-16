#include <vector>
#include <algorithm>

class Solution {
public:
    int minEatingSpeed(std::vector<int>& piles, int h) {
        int left = 1;
        int right = 0;
        
        // The maximum possible useful eating speed is the size of the largest pile
        for (int pile : piles) {
            right = std::max(right, pile);
        }
        
        int result = right;
        
        while (left <= right) {
            int k = left + (right - left) / 2;
            
            // Use long long to prevent integer overflow during summation
            long long hours_needed = 0;
            
            for (int pile : piles) {
                // Integer math equivalent of std::ceil(pile / k)
                hours_needed += (pile + k - 1) / k; 
            }
            
            // If we can finish within h hours, this speed is valid.
            // Record it, but try to find an even slower valid speed.
            if (hours_needed <= h) {
                result = k;
                right = k - 1;
            } 
            // If it takes too long, we MUST eat faster.
            else {
                left = k + 1;
            }
        }
        
        return result;
    }
};