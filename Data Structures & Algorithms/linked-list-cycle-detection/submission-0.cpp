/**
 * Definition for singly-linked list.
 * struct ListNode {
 * int val;
 * ListNode *next;
 * ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    bool hasCycle(ListNode *head) {
        // If the list is empty or has only one node, a cycle is impossible
        if (head == nullptr || head->next == nullptr) {
            return false;
        }
        
        ListNode *slow = head;
        ListNode *fast = head;
        
        // Fast moves 2 steps, so we must ensure the next two nodes exist
        while (fast != nullptr && fast->next != nullptr) {
            slow = slow->next;         // Tortoise moves 1 step
            fast = fast->next->next;   // Hare moves 2 steps
            
            // If they land on the exact same node in memory, it's a cycle
            if (slow == fast) {
                return true;
            }
        }
        
        // If fast reaches the end of the list, it's a straight line
        return false;
    }
};