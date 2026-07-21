class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        counts = {}
        max_len = 0
        left = 0
        max_f = 0
        
        for right in range(len(s)):
            counts[s[right]] = counts.get(s[right], 0) + 1
            max_f = max(max_f, counts[s[right]])
            while (right - left + 1) - max_f > k:
                counts[s[left]] -= 1
                left += 1
            max_len = max(max_len, right - left + 1)
            
        return max_len