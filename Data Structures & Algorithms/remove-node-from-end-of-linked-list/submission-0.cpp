/**
 * Definition for singly-linked list.
 * struct ListNode {
 * int val;
 * ListNode *next;
 * ListNode() : val(0), next(nullptr) {}
 * ListNode(int x) : val(x), next(nullptr) {}
 * ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        // Create a dummy node attached to the head
        // This makes edge cases (like deleting the head itself) trivial
        ListNode* dummy = new ListNode(0, head);
        ListNode* slow = dummy;
        ListNode* fast = dummy;
        
        // Step 1: Give 'fast' a head start of 'n' steps
        for (int i = 0; i < n; ++i) {
            fast = fast->next;
        }
        
        // Step 2: Move both pointers until 'fast' reaches the last node
        while (fast->next != nullptr) {
            slow = slow->next;
            fast = fast->next;
        }
        
        // Step 3: 'slow' is now positioned exactly before the node to delete
        ListNode* nodeToDelete = slow->next;
        slow->next = slow->next->next; // Bypass the target node
        
        // Clean up memory to prevent memory leaks
        delete nodeToDelete;
        
        // Store the new head before deleting the dummy node
        ListNode* newHead = dummy->next;
        delete dummy;
        
        return newHead;
    }
};