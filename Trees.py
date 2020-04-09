'''
https://www.geeksforgeeks.org/inorder-tree-traversal-without-recursion/

https://www.geeksforgeeks.org/check-if-a-given-binary-tree-is-heap/

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
    def inOrderTraverse(self, currentNode) :
        if currentNode is None :
            return
        self.inOrderTraverse(currentNode.left)
        print(currentNode.data)
        self.inOrderTraverse(currentNode.right)
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
heapSort(arr)
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
