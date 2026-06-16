#include <string>
#include <unordered_map>
#include <vector>

class TimeMap {
private:
    // Maps a string key to a vector of {timestamp, value} pairs
    std::unordered_map<std::string, std::vector<std::pair<int, std::string>>> store;

public:
    TimeMap() {
    }
    
    void set(std::string key, std::string value, int timestamp) {
        // Because timestamps are strictly increasing, push_back naturally keeps the vector sorted
        store[key].push_back({timestamp, value});
    }
    
    std::string get(std::string key, int timestamp) {
        // If the key doesn't exist, return empty string
        if (store.find(key) == store.end()) {
            return "";
        }
        
        const auto& values = store[key];
        int left = 0;
        int right = values.size() - 1;
        std::string result = "";
        
        // Binary search to find the largest timestamp <= target timestamp
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (values[mid].first <= timestamp) {
                // This is a valid candidate. Record it, but keep searching to the right
                // to see if there is an even closer (larger) valid timestamp.
                result = values[mid].second;
                left = mid + 1; 
            } else {
                // This timestamp is strictly too large. We must search to the left.
                right = mid - 1; 
            }
        }
        
        return result;
    }
};