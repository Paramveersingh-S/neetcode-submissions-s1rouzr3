#include <vector>
#include <algorithm>
#include <climits>

class Solution {
public:
    int maxProfit(std::vector<int>& prices) {
        int min_price = INT_MAX;
        int max_profit = 0;
        
        for (int price : prices) {
            // Keep track of the lowest price we've seen so far
            min_price = std::min(min_price, price);
            
            // Check if selling at the current price yields a better profit
            max_profit = std::max(max_profit, price - min_price);
        }
        
        return max_profit;
    }
};