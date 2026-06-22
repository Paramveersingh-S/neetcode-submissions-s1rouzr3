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
    ListNode* reverseKGroup(ListNode* head, int k) {
        // Edge cases: empty list, or groups of 1 which require no reversal
        if (head == nullptr || k == 1) {
            return head;
        }

        ListNode dummy(0);
        dummy.next = head;
        ListNode* groupPrev = &dummy;

        while (true) {
            // 1. Check if there are k nodes left to reverse
            ListNode* kth = getKth(groupPrev, k);
            if (kth == nullptr) {
                break; // Fewer than k nodes remain, leave them as they are
            }
            
            ListNode* groupNext = kth->next;
            
            // 2. Reverse the group
            // Initialize prev to groupNext so the tail automatically connects to the rest of the list
            ListNode* prev = groupNext; 
            ListNode* curr = groupPrev->next;
            
            while (curr != groupNext) {
                ListNode* temp = curr->next;
                curr->next = prev;
                prev = curr;
                curr = temp;
            }
            
            // 3. Stitch the reversed group back to the main chain
            ListNode* tmp = groupPrev->next; // The original head of the group is now the tail
            groupPrev->next = kth;           // Stitch the previous part to the new head of the group
            groupPrev = tmp;                 // Move groupPrev forward to the tail of this group
        }
        
        return dummy.next;
    }

private:
    // Helper function to safely jump forward k steps
    ListNode* getKth(ListNode* curr, int k) {
        while (curr != nullptr && k > 0) {
            curr = curr->next;
            k--;
        }
        return curr;
    }
};