#include <string>
#include <stack>

class Solution {
public:
    bool isValid(std::string s) {
        std::stack<char> st;
        
        for (char c : s) {
            // If it's an opening bracket, push to stack
            if (c == '(' || c == '{' || c == '[') {
                st.push(c);
            } 
            // If it's a closing bracket
            else {
                // If stack is empty, there is no matching open bracket
                if (st.empty()) {
                    return false;
                }
                
                char top = st.top();
                st.pop();
                
                // Check for mismatches
                if (c == ')' && top != '(') return false;
                if (c == '}' && top != '{') return false;
                if (c == ']' && top != '[') return false;
            }
        }
        
        // If the stack is not empty at the end, we have unclosed brackets
        return st.empty();
    }
};