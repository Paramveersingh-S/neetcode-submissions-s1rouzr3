class Solution:
    def trap(self, height: list[int]) -> int:
        if not height or len(height) < 3:
            return 0
        l, r = 0, len(height) - 1
        max_left, max_right = height[l], height[r]
        
        trapped_water = 0
        
        while l < r:
            if max_left < max_right:
                l += 1
                max_left = max(max_left, height[l])
                
                trapped_water += max_left - height[l]
                
            else:
                r -= 1
                max_right = max(max_right, height[r])
                trapped_water += max_right - height[r]
                
        return trapped_water