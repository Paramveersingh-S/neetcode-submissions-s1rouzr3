#include <string>
#include <unordered_set>
#include <algorithm>

class Solution {
public:
    int lengthOfLongestSubstring(std::string s) {
        std::unordered_set<char> window_chars;
        int left = 0;
        int max_length = 0;
        
        for (int right = 0; right < s.length(); ++right) {
            // If we find a duplicate, shrink the window from the left
            while (window_chars.find(s[right]) != window_chars.end()) {
                window_chars.erase(s[left]);
                left++;
            }
            
            // Add the new character to the window
            window_chars.insert(s[right]);
            
            // Update the maximum length found so far
            max_length = std::max(max_length, right - left + 1);
        }
        
        return max_length;
    }
};