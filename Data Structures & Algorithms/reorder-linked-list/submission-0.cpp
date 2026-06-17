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
    void reorderList(ListNode* head) {
        if (!head || !head->next) return;
        
        // Phase 1: Find the middle of the linked list
        ListNode* slow = head;
        ListNode* fast = head->next;
        
        while (fast != nullptr && fast->next != nullptr) {
            slow = slow->next;
            fast = fast->next->next;
        }
        
        // Phase 2: Reverse the second half of the list
        ListNode* second = slow->next;
        slow->next = nullptr; // Break the list into two separate chains
        ListNode* prev = nullptr;
        
        while (second != nullptr) {
            ListNode* temp = second->next;
            second->next = prev;
            prev = second;
            second = temp;
        }
        
        // Phase 3: Merge the two halves alternatingly
        ListNode* first = head;
        second = prev; // 'prev' is the new head of the reversed second half
        
        while (second != nullptr) {
            // Save the next nodes
            ListNode* temp1 = first->next;
            ListNode* temp2 = second->next;
            
            // Link first half node to second half node
            first->next = second;
            // Link second half node back to the next first half node
            second->next = temp1;
            
            // Step both pointers forward
            first = temp1;
            second = temp2;
        }
    }
};