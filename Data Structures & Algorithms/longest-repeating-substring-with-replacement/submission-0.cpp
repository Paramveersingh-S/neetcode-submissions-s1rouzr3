#include <string>
#include <vector>
#include <algorithm>

class Solution {
public:
    int characterReplacement(std::string s, int k) {
        std::vector<int> count(26, 0);
        int left = 0;
        int max_f = 0;
        int max_length = 0;
        
        for (int right = 0; right < s.length(); ++right) {
            count[s[right] - 'A']++;
            max_f = std::max(max_f, count[s[right] - 'A']);
            
            if ((right - left + 1) - max_f > k) {
                count[s[left] - 'A']--;
                left++;
            }
            
            max_length = std::max(max_length, right - left + 1);
        }
        
        return max_length;
    }
};