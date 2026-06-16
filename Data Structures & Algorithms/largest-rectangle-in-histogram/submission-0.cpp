#include <vector>
#include <stack>
#include <algorithm>

class Solution {
public:
    int largestRectangleArea(std::vector<int>& heights) {
        int max_area = 0;
        // Stack stores pairs: {start_index, height}
        std::stack<std::pair<int, int>> st;
        int n = heights.size();
        
        for (int i = 0; i < n; ++i) {
            int start = i;
            
            // If the current bar is shorter, it bottlenecks the taller bars in the stack
            while (!st.empty() && heights[i] < st.top().second) {
                auto [idx, h] = st.top();
                st.pop();
                
                // Calculate area for the popped bar
                max_area = std::max(max_area, h * (i - idx));
                
                // The current shorter bar can expand backward to the popped bar's starting index
                start = idx;
            }
            
            // Push the current bar with its new, furthest-left starting index
            st.push({start, heights[i]});
        }
        
        // Process any remaining bars that extend all the way to the right edge
        while (!st.empty()) {
            auto [idx, h] = st.top();
            st.pop();
            max_area = std::max(max_area, h * (n - idx));
        }
        
        return max_area;
    }
};