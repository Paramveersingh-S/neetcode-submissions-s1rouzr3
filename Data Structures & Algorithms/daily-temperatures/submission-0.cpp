#include <vector>
#include <stack>

class Solution {
public:
    std::vector<int> dailyTemperatures(std::vector<int>& temperatures) {
        int n = temperatures.size();
        std::vector<int> result(n, 0);
        std::stack<int> st; // This will store the indices of the temperatures
        
        for (int i = 0; i < n; ++i) {
            // While the stack is not empty AND the current day's temperature 
            // is warmer than the temperature at the index on the top of the stack
            while (!st.empty() && temperatures[i] > temperatures[st.top()]) {
                int prev_index = st.top();
                st.pop();
                
                // Calculate how many days we waited
                result[prev_index] = i - prev_index;
            }
            
            // Push the current day's index onto the stack to wait for a warmer day
            st.push(i);
        }
        
        return result;
    }
};