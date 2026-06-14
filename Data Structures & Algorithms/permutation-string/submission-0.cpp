#include <string>
#include <vector>

class Solution {
public:
    bool checkInclusion(std::string s1, std::string s2) {
        int len1 = s1.length();
        int len2 = s2.length();
        
        if (len1 > len2) {
            return false;
        }
        
        std::vector<int> s1_count(26, 0);
        std::vector<int> window_count(26, 0);
        
        // Populate the frequency of s1 and the first window of s2
        for (int i = 0; i < len1; ++i) {
            s1_count[s1[i] - 'a']++;
            window_count[s2[i] - 'a']++;
        }
        
        // Slide the window across s2
        for (int i = len1; i < len2; ++i) {
            // Check if the current window matches s1
            if (s1_count == window_count) {
                return true;
            }
            
            // Add the new character entering the window on the right
            window_count[s2[i] - 'a']++;
            
            // Remove the old character leaving the window from the left
            window_count[s2[i - len1] - 'a']--;
        }
        
        // Check the very last window after the loop finishes
        return s1_count == window_count;
    }
};