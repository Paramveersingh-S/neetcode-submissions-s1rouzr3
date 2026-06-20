#include <vector>

class Solution {
public:
    int findDuplicate(std::vector<int>& nums) {
        // Phase 1: Find the intersection point of the two runners.
        int slow = nums[0];
        int fast = nums[0];
        
        // We use a do-while loop because they both start at nums[0]
        do {
            slow = nums[slow];                 // Tortoise moves 1 step
            fast = nums[nums[fast]];           // Hare moves 2 steps
        } while (slow != fast);
        
        // Phase 2: Find the entrance to the cycle.
        int slow2 = nums[0];
        
        // Move both pointers at the same speed until they collide.
        while (slow2 != slow) {
            slow2 = nums[slow2];
            slow = nums[slow];
        }
        
        // The collision point is the duplicate number.
        return slow2;
    }
};