'''
https://www.geeksforgeeks.org/construct-tree-from-given-inorder-and-preorder-traversal/

https://www.geeksforgeeks.org/check-whether-binary-tree-full-binary-tree-not/

https://www.geeksforgeeks.org/check-binary-tree-contains-duplicate-subtrees-size-2/

https://www.geeksforgeeks.org/check-given-graph-tree/

https://www.geeksforgeeks.org/print-longest-leaf-leaf-path-binary-tree/

https://www.geeksforgeeks.org/print-path-root-given-node-binary-tree/

https://www.geeksforgeeks.org/find-sum-left-leaves-given-binary-tree/

https://www.geeksforgeeks.org/find-sum-right-leaves-given-binary-tree/

https://www.geeksforgeeks.org/sum-nodes-longest-path-root-leaf-node/

https://www.geeksforgeeks.org/remove-all-nodes-which-lie-on-a-path-having-sum-less-than-k/

https://www.geeksforgeeks.org/find-maximum-path-sum-two-leaves-binary-tree/

https://www.geeksforgeeks.org/sum-nodes-k-th-level-tree-represented-string/

https://www.geeksforgeeks.org/level-maximum-number-nodes/

https://www.geeksforgeeks.org/smallest-value-level-binary-tree/
'''

class Node :
    def __init__(self,data) :
        self.data = data
        self.right = None
        self.left = None

        self.heapifiedArray = []

class Tree :
    def __init__(self, data) :
        self.root = Node(data)
    def height(self, currentNode) :
        if currentNode == None :
            return 0
        return 1 + max(self.height(currentNode.right), self.height(currentNode.left))
    def isBalanced(self,currentNode):
        # a None tree is balanced
        if currentNode is None:
            return True
        return (self.isBalanced(currentNode.right) and self.isBalanced(currentNode.left) and
                abs(self.height(currentNode.left) - self.height(currentNode.right)) <= 1)
    def maxNode(self, currentNode) :
        if currentNode == None :
            return 0
        left = self.maxNode(currentNode.left)
        right = self.maxNode(currentNode.right)
        return max(left, right, currentNode.data)
    def levelTraverse(self, node, queue) :
        if node != None :
            queue.append(node)
        else :
            return None

        while queue :
            n = queue.pop(0)
            print(n.data)
            if n.left :
                queue.append(n.left)
            if n.right :
                queue.append(n.right)

    def sumOfHeight(self, currentNode, height, sum) :
        if height > 0 :
            sum = currentNode.data
            if currentNode.left != None and currentNode.right != None:
                return self.sumOfHeight(currentNode.left, height-1, sum) + self.sumOfHeight(currentNode.right, height-1, sum)
            if currentNode.right != None:
                return self.sumOfHeight(currentNode.right, height-1, sum)
            if currentNode.left != None:
                return self.sumOfHeight(currentNode.left, height-1, sum)
        return sum
    def heapSort(self) :
        arrLen = len(self.heapifiedArray)
        for i in range(arrLen, -1, -1) :
            self.heapify(self.heapifiedArrayarrLen, i)

        for i in range(arrLen-1, 0, -1) :
            arr[i], arr[0] = arr[0], arr[i]
            self.heapify(arr, i, 0)
    def heapify(self, arr, arrLen, index) :
        largest = index
        leftChild = 2*i+1
        rightChild = 2*i+2

        if leftChild < arrLen and arr[leftChild] > arr[index] :
            largest = leftChild
        if rightChild < arrLen and arr[rightChild] > arr[largest] :
            largest = leftChild

        if largest != index :
            arr[index], arr[largest] = arr[largest], arr[index]
            self.heapify(arr, arrLen, largest)
    def printLeaves(self, currentNode) :
        if currentNode.left == None and currentNode.right == None and currentNode != None  :
            print (currentNode.data)
        if currentNode.left :
            self.printLeaves(currentNode.left)
        if currentNode.right :
            self.printLeaves(currentNode.right)



'''
            1
        2       3
          4       5
         6         7
'''

arr = [12, 11, 13, 5, 7, 6]
#heapSort(arr)
n = len(arr)
print ("Heap Sortis")
for i in range(n):
    print (arr[i])


tree = Tree(1)
tree.root.left = Node(2)
tree.root.left.right = Node(4)
tree.root.left.right.left = Node(6)
tree.root.right = Node(3)
tree.root.right.right = Node(5)
tree.root.right.right.right = Node(7)


tree.height(tree.root)
print("Tree is balanced? ", tree.isBalanced(tree.root))
print("inorder traverse : ")
tree.inOrderTraverse(tree.root)
print("max data = ",tree.maxNode(tree.root))
print("print Tree levelwise : ")
tree.levelTraverse(tree.root,[])
print("-------")
print("print sum of all nodes up to height 3:", tree.root.data + tree.sumOfHeight(tree.root,2,0))#prevents double counting on the right side of the tree


print("List of Roots: ")
tree.printLeaves(tree.root)


#############################################################################################################################################

'''
    check if a bin tree is a heap:
    It should be a complete tree (i.e. all levels except last should be full). 
    Every node’s value should be greater than or equal to its child node (considering max-heap).
'''

class Node :
    def __init__(self,data) :
        self.data = data
        self.right = None
        self.left = None


class Tree :
    def __init__(self, data) :
        self.root = Node(data)
    def treeIsHeap(self,): 
        node_count = self.count_nodes(self.root) 
        if (self.treeIsComplete(self.root, 0, node_count) and self.heap_propert_util(self.root)): 
            return True
        else: 
            return False
        
    def treeIsComplete(self, root, index, node_count): 
        if root is None: 
            return True
        if index >= node_count: 
            return False
        return (self.treeIsComplete(root.left, 2 * index + 1, node_count) and self.treeIsComplete(root.right, 2 * index + 2, node_count)) 
        
    def count_nodes(self, root): 
        if root is None: 
            return 0
        else: 
            return (1 + self.count_nodes(root.left) + self.count_nodes(root.right)) 
      
    def heap_propert_util(self, root): 
      
        if (root.left is None and root.right is None): 
            return True
      
        if root.right is None: 
            return root.data >= root.left.data 
        else: 
            if (root.data >= root.left.data and root.data >= root.right.data): 
                return (self.heap_propert_util(root.left) and self.heap_propert_util(root.right)) 
            else: 
                return False

#Test cases            
tree = Tree(1)
tree.root.left = Node(2)
tree.root.left.right = Node(4)
tree.root.left.right.left = Node(6)
tree.root.right = Node(3)
tree.root.right.right = Node(5)
tree.root.right.right.right = Node(7)
print("Tree is a heap : ", tree.treeIsHeap())     

tree2 = Tree(10)

tree2.root.left = Node(8)
tree2.root.right = Node(7)        

tree2.root.left.left = Node(6)
tree2.root.left.right = Node(5)
tree2.root.right.left = Node(5) 
tree2.root.right.right = Node(6)

tree2.root.left.left.left = Node(4)
print("Tree is a heap : ", tree2.treeIsHeap())          
      
   
#############################################################################################################################################
'''
    in irder traverse of a tree using recursion and stack
'''

class Node :
    def __init__(self,data) :
        self.data = data
        self.right = None
        self.left = None

class Tree :
    def inOrderTraverse(self, currentNode) :
        if currentNode is None :
            return
        self.inOrderTraverse(currentNode.left)
        print(currentNode.data)
        self.inOrderTraverse(currentNode.right)
    def inorderTraverseStack(self,):
        current = self.root
        stack = []
        print("Stackwise inorder:")
        while True:
            if current != None:
                stack.append(current)
                current = current.left
            elif len(stack) > 0:
                current = stack.pop()
                print(current.data)
                
                current = current.right
            else:
                break

#Test Case
tree = Tree()
tree.root = Node(0)

tree.root.left = Node(1)
tree.root.right = Node(2)        

tree.root.left.left = Node(3)
tree.root.left.right = Node(4)
tree.root.right.left = Node(5) 
tree.root.right.right = Node(6)

tree.root.left.left.left = Node(7)
tree.root.right.left.right = Node(8)

tree.root.right.left.right.left = Node(9)
tree.root.right.left.right.right = Node(10)  

print("In order traverse using recursion : ", tree.inOrderTraverse(tree.root))
print("In order traverse using recursion : ", tree.inorderTraverseStack())

        
        
