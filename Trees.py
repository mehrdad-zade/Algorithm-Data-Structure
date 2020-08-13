'''
Fibonacci heap

https://www.geeksforgeeks.org/duplicate-subtree-in-binary-tree-set-2/

https://www.geeksforgeeks.org/print-path-root-given-node-binary-tree/

https://www.geeksforgeeks.org/sum-nodes-longest-path-root-leaf-node/

https://www.geeksforgeeks.org/remove-all-nodes-which-lie-on-a-path-having-sum-less-than-k/

https://www.geeksforgeeks.org/find-maximum-path-sum-two-leaves-binary-tree/

https://www.geeksforgeeks.org/sum-nodes-k-th-level-tree-represented-string/

https://www.geeksforgeeks.org/level-maximum-number-nodes/

https://www.geeksforgeeks.org/smallest-value-level-binary-tree/
'''

#1############################################################################################################################################
class Node :
    def __init__(self,data) :
        self.data = data
        self.right = None
        self.left = None

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

    def sumOfLevel(self, current, level):
        if current is None:
            return 0
        if level == 0:
            return current.data
        else:
            return self.sumOfLevel(current.left, level-1) + self.sumOfLevel(current.right, level-1)
    

    def printLeaves(self, currentNode) :
        if currentNode.left == None and currentNode.right == None and currentNode != None  :
            print (currentNode.data)
        if currentNode.left :
            self.printLeaves(currentNode.left)
        if currentNode.right :
            self.printLeaves(currentNode.right)
    
#test cases
'''
            1
        2       3
          4       5
         6         7
'''
 
    
tree = Tree(1)
tree.root.left = Node(2)
tree.root.left.right = Node(4)
tree.root.left.right.left = Node(6)
tree.root.right = Node(3)
tree.root.right.right = Node(5)
tree.root.right.right.right = Node(7)


print("Tree height is : ", tree.height(tree.root))

print("Tree is balanced? ", tree.isBalanced(tree.root))

print("levelTraverse = ", tree.levelTraverse(tree.root, []))

print("Sum of nodes at level 2 = ", tree.sumOfLevel(tree.root, 2))

print("List of leaves : ") 
tree.printLeaves(tree.root)

#2############################################################################################################################################

'''
heap sort

you have to manage an array based on a full tree concept. the tree has to be ordered such that the parent of each node is bigger than its children.
if the child is smaller you'll have to swap on the tree and then swap on the array too. once there is nothing else to swap then
swap the root with the last element. last element will be dropped from the tree and from the array since it's at the end of the 
array we should keep tracker to know which item is the last element on the array which hasn't already been sorted.
this way you'll get a ascending array
'''

def heapSort(arr):
    n = len(arr)
    #max heap
    for i in range(n,-1,-1):
        heapify(arr, i, n)
    
    #keep largest at the end
    for i in range(n-1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, 0, i)
    
    return arr


    
def heapify(arr, i, n):    
    largest_index = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[i] < arr[l]:
        largest_index = l
    if r < n and arr[largest_index] < arr[r]:
        largest_index = r
    if largest_index != i:
        arr[largest_index], arr[i] = arr[i], arr[largest_index]
        heapify(arr, largest_index, n)
    
    
#test cases       
arr = [9,2,5,3,8,10,6,5,4,7]
print(heapSort(arr))    



#3############################################################################################################################################
#sum of right leaves and right children

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        
class Tree:
    def __init__(self, data):
        self.root = Node(data)
        self.sum = 0

    def sumRightLeaves(self, current):        
        if current is None:
            return
        if current.right is not None:
            if current.right.right is None and current.right.left is None:
                self.sum += current.right.data
        self.sumRightLeaves(current.left)
        self.sumRightLeaves(current.right)  
        
    def sumRightChildren(self, current):
        if current is None:
            return
        self.sumRightChildren(current.left)
        if current.right is not None:
            self.sum += current.right.data
        self.sumRightChildren(current.right)        

             
        
    
#test case
tree = Tree(1)  
tree.root.left = Node(2)  
tree.root.left.left = Node(4)  
tree.root.left.right = Node(5)  
tree.root.left.left.right = Node(2)  
tree.root.right = Node(3)  
tree.root.right.right = Node(8)  
tree.root.right.right.left = Node(6)  
tree.root.right.right.right = Node(7) 

tree.sumRightLeaves(tree.root)    
print("Sum of right leaves = ", tree.sum) 

tree.sum = 0
tree.sumRightChildren(tree.root)
print("Sum of right Chilredn = ", tree.sum)


#4############################################################################################################################################

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
        
        #if right is none only compare left. else means we have a right so we expect a left too.
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
      
   
#5############################################################################################################################################
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

        
#6############################################################################################################################################        
'''
Construct Tree from given Inorder and Preorder traversals

Let us consider the below traversals:

Inorder sequence: D B E A F C
Preorder sequence: A B D E C F

In a Preorder sequence, leftmost element is the root of the tree. So we know ‘A’ is root for given sequences. By searching ‘A’ in Inorder sequence, we can find out all elements on left side of ‘A’ are in left subtree and elements on right are in right subtree. So we know below structure now.



                 A
               /   \
             /       \
           D B E     F C

We recursively follow above steps and get the following tree.

         A
       /   \
     /       \
    B         C
   / \        /
 /     \    /
D       E  F

           
'''        

class Node:
    def __init__(self, data):
        self.data = data
        self.right = None
        self.left = None


class Tree:
    def __init__(self,):
        self.root = None    
    def inorderTraverse(self, node):
        if node == None:
            return None
        self.inorderTraverse(node.left)
        print(node.data)
        self.inorderTraverse(node.right)
    def preorderTraverse(self, node):
        if node == None:
            return None
        print(node.data)
        self.preorderTraverse(node.left)        
        self.preorderTraverse(node.right)
    def buildTreeFromInorderAndPreorder(self, inorder, preorder):  
        return self.treeBuild_helper(0, 0, len(preorder)-1, inorder, preorder)
    def treeBuild_helper(self, preorderIndex, inorderStart, inorderEnd, inorder, preorder):
        if preorderIndex > len(preorder) - 1 or inorderStart > inorderEnd:
            return None
        
        root = Node(preorder[preorderIndex])
 
        inorderIndex = 0
        for i in range(len(inorder)):
            if inorder[i] == preorder[preorderIndex]:
                inorderIndex = i
                break
        
        root.left = self.treeBuild_helper(preorderIndex + 1, inorderStart, inorderIndex - 1, inorder, preorder)
        root.right = self.treeBuild_helper(preorderIndex + inorderIndex - inorderStart + 1, inorderIndex + 1, inorderEnd, inorder, preorder)

        return root        

     
        



#Test Case
tree = Tree()
inorder = [7, 3, 1, 4, 0, 5, 9, 8, 10, 2, 6]
preorder = [0, 1, 3, 7, 4, 2, 5, 8, 9, 10, 6]

print("Bin Tree from preorder and preorder traverse of this tree")
print("inorder = ", tree.inorderTraverse(tree.buildTreeFromInorderAndPreorder(inorder, preorder)))
#print("preorder = ", tree.preorderTraverse(tree.root))
        
#7############################################################################################################################################        

'''
find out if a tree is a full bin tree:
    1. if it's empty
    2. every node should have 0 or 2 childeren
'''

class Node:
    def __init__(self, data):
        self.data = data
        self.left, self.right = None, None

class Tree:
    def __init__(self,):
        self. root = None
    def isFullBinTree(self, root):
        if root == None:
            return True
        else:
            if root.left is not None and root.right is not None:
                return self.isFullBinTree(root.left) and self.isFullBinTree(root.right)   
            elif root.left == None and root.right == None:
                return True
            else:
                return False

#Test Case
#you can remove 100 to see the full bin true
tree = Tree()
tree.root = Node(0)


tree.root.left = Node(1)            
tree.root.right = Node(2)

tree.root.left.left = Node(3)            
tree.root.left.right = Node(4)    

tree.root.right.left = Node(100)                    


print("Is tree a Full Bin Tree? ", tree.isFullBinTree(tree.root))

#8############################################################################################################################################        

'''
Print the longest leaf-to-leaf path in a Binary tree. 

The diameter of a tree (width) is the number of nodes on the longest path 
between two end nodes.

We know that Diameter of a tree can be calculated by only using the height 
function because the diameter of a tree is nothing but the maximum value of 
(left_height + right_height + 1) for each node.
Now for the node which has the maximum value of (left_height + right_height + 1), 
we find the longest root to leaf path on the left side and similarly on the 
right side. Finally, we print left side path, root and right side path. Time 
Complexity is O(N). N is the number of nodes in the tree.

'''

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        
class Tree:
    def __init__(self, data):
        self.root = Node(data)
    def getDiameter_util(self, current, diameter):
        if current is None:
            return 0, diameter
        left_height, diameter = self.getDiameter_util(current.left, diameter)
        right_height, diameter = self.getDiameter_util(current.right, diameter)      
        print("diameter=", diameter)
        print("left_height=", left_height)
        print("right_height=", right_height)
        print("----------------------------")
        maxDiameter = left_height + right_height + 1
        diameter = max(maxDiameter, diameter)
        return max(left_height, right_height) + 1, diameter
        
    def getDiameter(self, root):
        diameter = 0
        return self.getDiameter_util(root, diameter)[1]
    
#test case
'''       
        1
       / \
      2   3
    /  \
  4     5
   \   / \
    8 6   7
   /
  9
 '''     
tree = Tree(1)
tree.root.left = Node(2); 
tree.root.right = Node(3); 
tree.root.left.left = Node(4); 
tree.root.left.right = Node(5); 
tree.root.left.right.left = Node(6); 
tree.root.left.right.right = Node(7); 
tree.root.left.left.right = Node(8); 
tree.root.left.left.right.left = Node(9); 
print("Tree diameter = ", tree.getDiameter(tree.root))   

#9############################################################################################################################################        

'''
Q1. Find sum of all left leaves in a given Binary Tree
Q2. Find sum of all left nodes  in a given Binary Tree
'''

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        
class Tree:
    def __init__(self, data):
        self.root = Node(data)      
        
    def isLeaf(self, node):
        if node == None:
            return False
        if node.left == None and node.right == None:
            return True
        return False
    
    def leftLeafSum(self, node):
        sum = 0
        if node is not None:
            if self.isLeaf(node.left):
                sum += node.left.data
            else:
                sum += self.leftLeafSum(node.left)
            sum += self.leftLeafSum(node.right)
        return sum
    def leftNodeSum(self, node):
        sum = 0
        if node is not None:
            if node.left is not None:
                sum += node.left.data
            sum += self.leftNodeSum(node.left)
            sum += self.leftNodeSum(node.right)
        return sum
            
    
#test case
'''
        20
      /    \
     9      49
    / \     / \
   5   12  23  52
         \      / \
          12   50
'''        
    
tree = Tree(20)
tree.root.left = Node(9) 
tree.root.right = Node(49) 
tree.root.right.left = Node(23)         
tree.root.right.right = Node(52) 
tree.root.right.right.left = Node(50) 
tree.root.left.left = Node(5) 
tree.root.left.right = Node(12) 
tree.root.left.right.right = Node(12) 
print ("Sum of left leaves is ", tree.leftLeafSum(tree.root))
print ("Sum of left nodes is ", tree.leftNodeSum(tree.root))

#############################################################################################################################################        

