#include <vector>
#include <algorithm>

class MinStack {
private:
    // pair.first = actual value
    // pair.second = minimum value up to this depth
    std::vector<std::pair<int, int>> st;

public:
    MinStack() {}
    
    void push(int val) {
        if (st.empty()) {
            st.push_back({val, val});
        } else {
            // Calculate the minimum between the new value and the previous minimum
            st.push_back({val, std::min(val, st.back().second)});
        }
    }
    
    void pop() {
        st.pop_back();
    }
    
    int top() {
        return st.back().first;
    }
    
    int getMin() {
        return st.back().second;
    }
};