#include <vector>
#include <algorithm>
#include <stack>

class Solution {
public:
    int carFleet(int target, std::vector<int>& position, std::vector<int>& speed) {
        int n = position.size();
        std::vector<std::pair<int, int>> cars(n);
        
        for (int i = 0; i < n; ++i) {
            cars[i] = {position[i], speed[i]};
        }
        
        // Sort cars by position descending (closest to the target first)
        std::sort(cars.rbegin(), cars.rend());
        
        std::stack<double> fleets;
        
        for (int i = 0; i < n; ++i) {
            double time = (double)(target - cars[i].first) / cars[i].second;
            
            // If the stack is empty, or this car takes longer than the fleet ahead of it
            // It forms a new, slower fleet.
            if (fleets.empty() || time > fleets.top()) {
                fleets.push(time);
            }
            // Otherwise, it takes less or equal time, meaning it catches up to the fleet 
            // ahead. We do nothing, effectively merging it into the top fleet.
        }
        
        return fleets.size();
    }
};