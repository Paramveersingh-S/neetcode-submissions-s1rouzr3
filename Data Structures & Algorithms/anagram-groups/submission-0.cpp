#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

class Solution {
public:
    std::vector<std::vector<std::string>> groupAnagrams(std::vector<std::string>& strs) {
        std::unordered_map<std::string, std::vector<std::string>> anagramMap;
        
        for (const std::string& str : strs) {
            std::string key = str;
            std::sort(key.begin(), key.end());
            anagramMap[key].push_back(str);
        }
        
        std::vector<std::vector<std::string>> result;
        result.reserve(anagramMap.size());
        
        for (auto& pair : anagramMap) {
            result.push_back(std::move(pair.second));
        }
        
        return result;
    }
};