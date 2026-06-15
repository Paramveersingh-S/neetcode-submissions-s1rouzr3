#include <vector>
#include <string>
#include <stack>

class Solution {
public:
    int evalRPN(std::vector<std::string>& tokens) {
        std::stack<int> st;
        
        for (const std::string& token : tokens) {
            // Check if the token is an operator
            if (token == "+" || token == "-" || token == "*" || token == "/") {
                // The first popped element is the right operand (b)
                int b = st.top(); 
                st.pop();
                
                // The second popped element is the left operand (a)
                int a = st.top(); 
                st.pop();
                
                // Evaluate and push the result back
                if (token == "+") st.push(a + b);
                else if (token == "-") st.push(a - b);
                else if (token == "*") st.push(a * b);
                else if (token == "/") st.push(a / b);
            } else {
                // If it's a number, convert string to integer and push
                st.push(std::stoi(token));
            }
        }
        
        // The final result is the only item left in the stack
        return st.top();
    }
};