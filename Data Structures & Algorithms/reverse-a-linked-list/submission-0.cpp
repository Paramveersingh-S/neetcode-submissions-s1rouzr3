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
    ListNode* reverseList(ListNode* head) {
        ListNode* prev = nullptr;
        ListNode* curr = head;
        ListNode* next = nullptr;
        
        while (curr != nullptr) {
            // Save the next node
            next = curr->next;
            
            // Reverse the pointer
            curr->next = prev;
            
            // Step both pointers forward
            prev = curr;
            curr = next;
        }
        
        // prev will be the new head
        return prev;
    }
};