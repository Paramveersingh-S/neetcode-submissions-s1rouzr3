class MinStack:
    def __init__(self):
        # We will store tuples in the format: (value, min_so_far)
        self.stack = []

    def push(self, val: int) -> None:
        # If the stack is empty, the current value is obviously the minimum
        if not self.stack:
            self.stack.append((val, val))
        else:
            # Look at the minimum of the previous top element
            previous_min = self.stack[-1][1]
            
            # The new minimum is whichever is smaller: our new value, or the previous minimum
            current_min = min(val, previous_min)
            
            # Push the pair onto the stack
            self.stack.append((val, current_min))

    def pop(self) -> None:
        # Python's built-in pop() removes the last item from the list in O(1) time
        self.stack.pop()

    def top(self) -> int:
        # Return the 0th index of the tuple at the top of the stack (the actual value)
        return self.stack[-1][0]

    def getMin(self) -> int:
        # Return the 1st index of the tuple at the top of the stack (the historical minimum)
        return self.stack[-1][1]