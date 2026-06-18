/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* next;
    Node* random;
    
    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};
*/

class Solution {
public:
    Node* copyRandomList(Node* head) {
        if (head == nullptr) {
            return nullptr;
        }
        
        // Phase 1: Clone nodes and weave them into the original list
        Node* curr = head;
        while (curr != nullptr) {
            Node* copy = new Node(curr->val);
            copy->next = curr->next;
            curr->next = copy;
            curr = copy->next;
        }
        
        // Phase 2: Assign random pointers for the copied nodes
        curr = head;
        while (curr != nullptr) {
            // Check if the original node actually has a random pointer
            if (curr->random != nullptr) {
                // The copy's random is the original random's next (which is its copy)
                curr->next->random = curr->random->next;
            }
            curr = curr->next->next; // Skip over the copy to the next original node
        }
        
        // Phase 3: Unweave the lists (restore original, extract the copy)
        curr = head;
        Node* dummy = new Node(0); // Dummy node to anchor the copied list
        Node* copyTail = dummy;
        
        while (curr != nullptr) {
            // Save the next original node before we break the links
            Node* nextOrig = curr->next->next;
            
            // Extract the copied node and attach it to our new chain
            copyTail->next = curr->next;
            copyTail = copyTail->next;
            
            // Restore the original list's 'next' pointer
            curr->next = nextOrig;
            
            // Step forward
            curr = nextOrig;
        }
        
        // Save the real head of the copied list and clean up the dummy node
        Node* copyHead = dummy->next;
        delete dummy; 
        
        return copyHead;
    }
};