#include <vector>
#include <queue>

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
    // We must define a custom comparator for our priority queue
    // because it needs to know how to compare two ListNode objects.
    struct CompareNode {
        bool operator()(ListNode* a, ListNode* b) {
            // Use > to create a Min-Heap (smallest values bubble to the top)
            return a->val > b->val; 
        }
    };

    ListNode* mergeKLists(std::vector<ListNode*>& lists) {
        // Initialize the priority queue with our custom comparator
        std::priority_queue<ListNode*, std::vector<ListNode*>, CompareNode> minHeap;
        
        // Phase 1: Push the head of every non-empty list into the heap
        for (ListNode* head : lists) {
            if (head != nullptr) {
                minHeap.push(head);
            }
        }
        
        // Setup the result list
        ListNode dummy(0);
        ListNode* tail = &dummy;
        
        // Phase 2: Process the heap
        while (!minHeap.empty()) {
            // Extract the absolute smallest node currently available
            ListNode* smallest = minHeap.top();
            minHeap.pop();
            
            // Attach it to our merged list
            tail->next = smallest;
            tail = tail->next;
            
            // If the list we just pulled from has more nodes, 
            // push its next node into the heap to act as its new representative.
            if (smallest->next != nullptr) {
                minHeap.push(smallest->next);
            }
        }
        
        return dummy.next;
    }
};