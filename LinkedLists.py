class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    def push(self, val):
        if self.head == None:
            self.head = Node(val)
        else:
            current = self.head
            while current.next != None:
                current = current.next
            current.next = Node(val)
    def length(self):
        if self.head == None:
            return 0
        current = self.head
        count = 0
        while current != None:
            count += 1
            current = current.next
        return count


#Test Cases
l = LinkedList()
#insert nodes by tapping into the class variables
l.head = Node(1)
l.head.next = Node(2)
l.head.next.next = Node(3)
l.head.next.next.next = Node(4)
#a neat way of putting values at the end of a linkedlist
l.push(5)
l.push(6)
l.push(7)
print("Linked List Length = ", l.length())

###############################################################################
