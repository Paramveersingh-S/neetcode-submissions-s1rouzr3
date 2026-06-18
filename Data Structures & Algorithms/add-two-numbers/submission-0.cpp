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
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        // Dummy node to easily return the head of the new list
        ListNode* dummy = new ListNode(0);
        ListNode* curr = dummy;
        int carry = 0;
        
        // Loop while there are nodes left in either list, or if there's a leftover carry
        while (l1 != nullptr || l2 != nullptr || carry != 0) {
            int sum = carry;
            
            // Add l1's value if it exists, then step forward
            if (l1 != nullptr) {
                sum += l1->val;
                l1 = l1->next;
            }
            
            // Add l2's value if it exists, then step forward
            if (l2 != nullptr) {
                sum += l2->val;
                l2 = l2->next;
            }
            
            // Extract the new carry (e.g., 18 / 10 = 1)
            carry = sum / 10;
            
            // Extract the actual digit (e.g., 18 % 10 = 8) and attach it
            curr->next = new ListNode(sum % 10);
            
            // Move our result pointer forward
            curr = curr->next;
        }
        
        ListNode* resultHead = dummy->next;
        delete dummy; // Clean up memory
        
        return resultHead;
    }
};