#include <unordered_map>

// Doubly Linked List Node
class Node {
public:
    int key;
    int val;
    Node* prev;
    Node* next;
    
    Node(int k, int v) {
        key = k;
        val = v;
        prev = nullptr;
        next = nullptr;
    }
};

class LRUCache {
private:
    int capacity;
    std::unordered_map<int, Node*> cache;
    Node* head;
    Node* tail;
    
    // Helper: Always add the new node right after the dummy head (MRU position)
    void addNode(Node* node) {
        Node* temp = head->next;
        head->next = node;
        node->prev = head;
        node->next = temp;
        temp->prev = node;
    }
    
    // Helper: Pluck an existing node out of the linked list
    void removeNode(Node* node) {
        Node* prevNode = node->prev;
        Node* nextNode = node->next;
        prevNode->next = nextNode;
        nextNode->prev = prevNode;
    }

public:
    LRUCache(int cap) {
        capacity = cap;
        // Setup dummy head and tail
        head = new Node(-1, -1);
        tail = new Node(-1, -1);
        head->next = tail;
        tail->prev = head;
    }
    
    int get(int key) {
        if (cache.find(key) != cache.end()) {
            Node* resNode = cache[key];
            int ans = resNode->val;
            
            // Mark as recently used: remove from current spot, add to front
            removeNode(resNode);
            addNode(resNode);
            
            return ans;
        }
        return -1;
    }
    
    void put(int key, int value) {
        // If the key already exists, update its value and move to front
        if (cache.find(key) != cache.end()) {
            Node* existingNode = cache[key];
            existingNode->val = value;
            removeNode(existingNode);
            addNode(existingNode);
        } else {
            // If cache is full, we must evict the LRU node (right before tail)
            if (cache.size() == capacity) {
                Node* lruNode = tail->prev;
                cache.erase(lruNode->key); // Delete from Hash Map
                removeNode(lruNode);       // Delete from Linked List
                delete lruNode;            // Free memory
            }
            
            // Create the new node, add to Hash Map, add to front of list
            Node* newNode = new Node(key, value);
            cache[key] = newNode;
            addNode(newNode);
        }
    }
};