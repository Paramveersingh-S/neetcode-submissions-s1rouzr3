#include <vector>
#include <string>

class Solution {
public:

    std::string encode(std::vector<std::string>& strs) {
        std::string encoded = "";
        for (const std::string& str : strs) {
            encoded += std::to_string(str.length()) + "#" + str;
        }
        return encoded;
    }

    std::vector<std::string> decode(std::string s) {
        std::vector<std::string> decoded;
        int i = 0;
        
        while (i < s.length()) {
            int j = i;
            while (s[j] != '#') {
                j++;
            }
            
            int length = std::stoi(s.substr(i, j - i));
            decoded.push_back(s.substr(j + 1, length));
            
            i = j + 1 + length;
        }
        
        return decoded;
    }
};