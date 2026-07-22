class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if t == "" or len(t) > len(s):
            return ""
            
        count_t = {}
        window = {}
        
        for char in t:
            count_t[char] = count_t.get(char, 0) + 1
        have, need = 0, len(count_t)
        res, res_len = [-1, -1], float("inf")
        l = 0
        
        for r in range(len(s)):
            char = s[r]
            window[char] = window.get(char, 0) + 1
            if char in count_t and window[char] == count_t[char]:
                have += 1
            while have == need:
                if (r - l + 1) < res_len:
                    res = [l, r]
                    res_len = r - l + 1
                left_char = s[l]
                window[left_char] -= 1
                if left_char in count_t and window[left_char] < count_t[left_char]:
                    have -= 1
                l += 1
        l_idx, r_idx = res
        return s[l_idx : r_idx + 1] if res_len != float("inf") else ""