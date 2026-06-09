#include <vector>

class Solution {
public:
    bool isValidSudoku(std::vector<std::vector<char>>& board) {
        // boolean arrays initialized to false
        bool rows[9][9] = {false};
        bool cols[9][9] = {false};
        bool boxes[9][9] = {false};
        
        for (int r = 0; r < 9; ++r) {
            for (int c = 0; c < 9; ++c) {
                if (board[r][c] == '.') {
                    continue;
                }
                
                // Convert char '1'-'9' to integer 0-8
                int num = board[r][c] - '1'; 
                
                // Calculate which of the 9 boxes we are currently in
                int box_index = (r / 3) * 3 + (c / 3);
                
                // If we have already seen this number in the current row, col, or box
                if (rows[r][num] || cols[c][num] || boxes[box_index][num]) {
                    return false; 
                }
                
                // Mark the number as seen
                rows[r][num] = true;
                cols[c][num] = true;
                boxes[box_index][num] = true;
            }
        }
        
        return true;
    }
};