#include <string>
#include <vector>
#include <climits>

class Solution {
public:
    std::string minWindow(std::string s, std::string t) {
        if (s.empty() || t.empty() || s.length() < t.length()) {
            return "";
        }
        
        // Array to act as a frequency map for the 128 ASCII characters
        std::vector<int> map(128, 0);
        for (char c : t) {
            map[c]++;
        }
        
        int left = 0, right = 0;
        int min_start = 0, min_length = INT_MAX;
        int counter = t.length(); // Total characters we need to match
        
        while (right < s.length()) {
            // If the character at 'right' is needed, decrease the counter
            if (map[s[right]] > 0) {
                counter--;
            }
            // Decrease the required frequency (surplus characters will become negative)
            map[s[right]]--;
            right++; // Expand the window
            
            // When our window contains all required characters
            while (counter == 0) {
                // Record the minimum window seen so far
                if (right - left < min_length) {
                    min_start = left;
                    min_length = right - left;
                }
                
                // Remove the character at 'left' from the window
                map[s[left]]++;
                
                // If removing it creates a deficit, increment the counter to break the loop
                if (map[s[left]] > 0) {
                    counter++;
                }
                left++; // Shrink the window
            }
        }
        
        return min_length == INT_MAX ? "" : s.substr(min_start, min_length);
    }
};