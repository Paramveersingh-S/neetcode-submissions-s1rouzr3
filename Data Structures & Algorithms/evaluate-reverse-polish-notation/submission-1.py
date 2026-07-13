class Solution:
    def evalRPN(self, tokens: list[str]) -> int:
        stack = []
        
        for token in tokens:
            if token == "+":
                # Pop right operand first, then left operand
                right = stack.pop()
                left = stack.pop()
                stack.append(left + right)
                
            elif token == "-":
                right = stack.pop()
                left = stack.pop()
                stack.append(left - right)
                
            elif token == "*":
                right = stack.pop()
                left = stack.pop()
                stack.append(left * right)
                
            elif token == "/":
                right = stack.pop()
                left = stack.pop()
                # Use int(left / right) to truncate toward zero, avoiding Python's // floor behavior
                stack.append(int(left / right))
                
            else:
                # If it's not an operator, it must be a number. Convert to int and push.
                stack.append(int(token))
                
        # The final result will be the only number left in the stack
        return stack[0]