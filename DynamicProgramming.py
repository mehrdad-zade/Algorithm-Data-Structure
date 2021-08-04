#0#########################
'''
fibo with DP

we need to memoise the previous fibos. for instance:

             fibo(5)
            /       \
        fibo(4)     fibo(3)
        /     \         /  \
    fibo(3)  fibo(2)   f(2) f(1)
    /   \       / \     /
    f(2) f(1)  f(1) f(1)...

as you can see a lot of f(x's) are repeated. and instead if recalculating them
we can memoise them to be reused in the recursion.

O(n) -> DP
O(n^2) -> without DP

'''

def fibo(n):
    memo = [None] * n
    return fibo_util(n, memo)

def fibo_util(n, memo):
    if memo[n-1] != None:
        return memo[n-1]
    if n == 1 or n == 2:
        return 1

    result = fibo(n-1) + fibo(n-2)
    memo[n-1] = result

    return result

print(fibo(10))        


#1##########################################################################################

"""

Collect maximum points in a grid using two traversals
Difficulty Level : Hard
 Last Updated : 27 Mar, 2019
Given a matrix where every cell represents points. How to collect maximum points using two traversals 
under following conditions?

Let the dimensions of given grid be R x C.

1) The first traversal starts from top left corner, i.e., (0, 0) and should reach left bottom corner, 
i.e., (R-1, 0). The second traversal starts from top right corner, i.e., (0, C-1) and should reach bottom 
right corner, i.e., (R-1, C-1)/

2) From a point (i, j), we can move to (i+1, j+1) or (i+1, j-1) or (i+1, j)

3) A traversal gets all points of a particular cell through which it passes. If one traversal has already 
collected points of a cell, then the other traversal gets no points if goes through that cell again.


Input :
    int arr[R][C] = {{3, 6, 8, 2},
                     {5, 2, 4, 3},
                     {1, 1, 20, 10},
                     {1, 1, 20, 10},
                     {1, 1, 20, 10},
                    };

     Output: 73

Explanation :
runninggrid
First traversal collects total points of value 3 + 2 + 20 + 1 + 1 = 27

Second traversal collects total points of value 2 + 4 + 10 + 20 + 10 = 46.
Total Points collected = 27 + 46 = 73.
"""
intmin = -10000000
intmax =  10000000

R = 5
C = 4

#check bounderies
def isValid(x,y1,y2): 
    return ((x >= 0 and x < R and y1 >=0 and y1 < C and y2 >=0 and y2 < C)) 

def gridMaxPoints(arr):
    # Create a memoization table and  initialize all entries as -1 
    # for each row we keep track of 2 Y's. player 1 and 2 x is always same, Y's are diff
    mem=[[[-1 for i in range(C)] for i in range(C)] for i in range(R)] 
      
    # Calculation maximum value using 
    return gridMaxPoints_util(arr, mem, 0, 0, C-1) 

def gridMaxPoints_util(arr, mem, x, y1, y2):
    # ---------- BASE CASES --------------------------------------------------
    if isValid(x, y1, y2) == False:
        return intmin
    
        # if both traversals reach their destinations
    if x == R-1 and y1 == 0 and y2 == C-1:
        if y1 == y2: 
            return arr[x][y1] 
        else: 
            return arr[x][y1]+arr[x][y2]
        
        # If both traversals are at last row  but not at their destination
    if x == R-1:
        return intmin
    
         # If subproblem is already solved 
    if mem[x][y1][y2] != -1:
        return mem[x][y1][y2]
     
    
    # ---------- Solve the subproblem ----------------------------------------

        # Initialize answer for this subproblem 
    ans=intmin 
  
        # this variable is used to store gain of current cell(s) 
    temp=0
    if y1==y2: 
        temp=arr[x][y1] 
    else: 
        temp=arr[x][y1]+arr[x][y2] 
        
        # Recur for all possible cases (3 possible positions for each player's move), 
        #then store and return the one with max value 
    ans = max(ans, temp + gridMaxPoints_util(arr, mem, x+1, y1, y2-1)) 
    ans = max(ans, temp + gridMaxPoints_util(arr, mem, x+1, y1, y2+1)) 
    ans = max(ans, temp + gridMaxPoints_util(arr, mem, x+1, y1, y2)) 
  
    ans = max(ans, temp + gridMaxPoints_util(arr, mem, x+1, y1-1, y2)) 
    ans = max(ans, temp + gridMaxPoints_util(arr, mem, x+1, y1-1, y2-1)) 
    ans = max(ans, temp + gridMaxPoints_util(arr, mem, x+1, y1-1, y2+1)) 
  
    ans = max(ans, temp + gridMaxPoints_util(arr, mem, x+1, y1+1, y2)) 
    ans = max(ans, temp + gridMaxPoints_util(arr, mem, x+1, y1+1, y2-1)) 
    ans = max(ans, temp + gridMaxPoints_util(arr, mem, x+1, y1+1, y2+1)) 
  
    mem[x][y1][y2] = ans 
    return ans 


if __name__=='__main__': 
    arr=[[3, 6, 8, 2], 
        [5, 2, 4, 3], 
        [1, 1, 20, 10], 
        [1, 1, 20, 10], 
        [1, 1, 20, 10], 
        ] 
    print('Maximum collection is ', gridMaxPoints(arr)) 

#2##########################################################################################

"""
How to print maximum number of A’s using given four keys

Imagine you have a special keyboard with the following keys: 
Key 1:  Prints 'A' on screen
Key 2: (Ctrl-A): Select screen
Key 3: (Ctrl-C): Copy selection to buffer
Key 4: (Ctrl-V): Print buffer on screen appending it
                 after what has already been printed. 

If you can only press the keyboard for N times (with the above four
keys), write a program to produce maximum numbers of A's. That is to
say, the input parameter is N (No. of keys that you can press), the 
output is M (No. of As that you can produce).

Example:
Input:  N = 3
Output: 3
We can at most get 3 A's on screen by pressing 
following key sequence.
A, A, A
"""
def findoptimal(N): 
      
    # The optimal string length is  
    # N when N is smaller than 6
    if N<= 6: 
        return N 
  
    # Initialize result  
    maxi = 0
  
    # TRY ALL POSSIBLE BREAK-POINTS  
    # For any keystroke N, we need 
    # to loop from N-3 keystrokes  
    # back to 1 keystroke to find  
    # a breakpoint  after which we  
    # will have Ctrl-A, Ctrl-C and then  
    # only Ctrl-V all the way. 
    for breakpoint in range(N-3, 0, -1): 
        curr =(N-breakpoint-1)*findoptimal(breakpoint) 
        if curr>maxi: 
            maxi = curr 
      
    return maxi 
# Driver program 
if __name__=='__main__': 
      
  
# for the rest of the array we will 
# rely on the previous  
# entries to compute new ones  
    for n in range(1, 21): 
        print('Maximum Number of As with ', n, 'keystrokes is ', findoptimal(n)) 
         
# this code is contibuted by sahilsh    

#3#############################################################################
'''
Given two strings str1 and str2 and below operations that can performed on str1. 
Find minimum number of edits (operations) required to convert ‘str1’ into ‘str2’.

Insert
Remove
Replace

Input:   str1 = "geek", str2 = "gesek"
Output:  1
We can convert str1 into str2 by inserting a 's'.

Dynamic Programming 

'''
def editDistDP(str1, str2, m, n): 
    # Create a table to store results of subproblems 
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)] 
  
    # Fill d[][] in bottom up manner 
    for i in range(m + 1): 
        for j in range(n + 1): 
  
            # If first string is empty, only option is to 
            # insert all characters of second string 
            if i == 0: 
                dp[i][j] = j    # Min. operations = j 
  
            # If second string is empty, only option is to 
            # remove all characters of second string 
            elif j == 0: 
                dp[i][j] = i    # Min. operations = i 
  
            # If last characters are same, ignore last char 
            # and recur for remaining string 
            elif str1[i-1] == str2[j-1]: 
                dp[i][j] = dp[i-1][j-1] 
  
            # If last character are different, consider all 
            # possibilities and find minimum 
            else: 
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert 
                                   dp[i-1][j],        # Remove 
                                   dp[i-1][j-1])    # Replace 
    print(dp)
    return dp[m][n] 
  
# test case 
str1 = "mehrdad"
str2 = "m"

print(editDistDP(str1, str2, len(str1), len(str2))) 

#4########################################################################################
'''
0-1 Knapsack Problem | DP-10

Given weights and values of n items, put these items in a knapsack of capacity W to 
get the maximum total value in the knapsack. In other words, given two integer arrays 
val[0..n-1] and wt[0..n-1] which represent values and weights associated with n items 
respectively. Also given an integer W which represents knapsack capacity, find out the 
maximum value subset of val[] such that sum of the weights of this subset is smaller 
than or equal to W. You cannot break an item, either pick the complete item or don’t 
pick it (0-1 property).

recursive solution : Time Complexity: O(2^n).
dynamic solution : Time Complexity: O(nw).
'''

def knapSack_recursive(value, weight, availableSize, numberOfItems):
    if numberOfItems <= 0 or availableSize <= 0: 
        return 0
    
    if weight[numberOfItems-1] > availableSize:
        return knapSack_recursive(value, weight, availableSize, numberOfItems-1)
    
    else:
        #nth item included or not
        return max( 
            value[numberOfItems-1] + knapSack_recursive( 
                value, weight, availableSize-weight[numberOfItems-1], numberOfItems-1),  
                knapSack_recursive(value, weight, availableSize, numberOfItems-1)) 
            
            
            
            
def knapSack_dynamic(value, weight, availableSize, numberOfItems):
    k = [[0 for i in range(availableSize+1)] for j in range(numberOfItems+1)]
    
    for i in range(numberOfItems+1):
        for w in range(availableSize+1):
            if i == 0 or w == 0:
                k[i][w] = 0
            elif weight[i-1] <= w:
                k[i][w] = max(value[i-1] + k[i-1][w-weight[i-1]] , k[i-1][w])
            else:
                k[i][w] = k[i-1][w]
                
    return k[n][w]
    
    
    
val = [60, 100, 120] 
wt = [10, 20, 30] 
W = 49
n = len(val) 
print("Knapsack Recursize answer : ", knapSack_recursive(val, wt, W, n))
print("Knapsack Dynamic answer : ", knapSack_dynamic(val, wt, W, n))

#5##########################################################################################

'''
Partition problem

Partition problem is to determine whether a given set can be partitioned into 
two subsets such that the sum of elements in both subsets is same.

Following are the two main steps to solve this problem:
1) Calculate sum of the array. If sum is odd, there can not be two subsets with 
    equal sum, so return false.
2) If sum of array elements is even, calculate sum/2 and find a subset of array 
    with sum equal to sum/2.

The first step is simple. The second step is crucial, it can be solved either 
using recursion or Dynamic Programming.

Recursive Solution
Following is the recursive property of the second step mentioned above.

Let isSubsetSum(arr, n, sum/2) be the function that returns true if 
there is a subset of arr[0..n-1] with sum equal to sum/2

The isSubsetSum problem can be divided into two subproblems
 a) isSubsetSum() without considering last element 
    (reducing n to n-1)
 b) isSubsetSum considering the last element 
    (reducing sum/2 by arr[n-1] and n to n-1)
If any of the above the above subproblems return true, then return true. 
isSubsetSum (arr, n, sum/2) = isSubsetSum (arr, n-1, sum/2) ||
                              isSubsetSum (arr, n-1, sum/2 - arr[n-1])
                              
Dynamic Programming Solution
The problem can be solved using dynamic programming when the sum of the elements is not too big. We can create a 2D array part[][] of size (sum/2 + 1)*(n+1). And we can construct the solution in bottom up manner such that every filled entry has following property

part[i][j] = true if a subset of {arr[0], arr[1], ..arr[j-1]} has sum 
             equal to i, otherwise false                              
                              

recursive solution : Time Complexity: O(2^n)
dynamic solution : Time Complexity: O(sum*n)
'''


def partition(arr):
    _sum = sum(arr)
    if _sum % 2 != 0:
        return False
    return isSubsetSum(arr, len(arr), _sum // 2)

def isSubsetSum(arr, n, _sum):
    if _sum == 0:
        return True
    #break point for the recursion to prevent index out of range
    if n == 0 and sum != 0: 
        return False
    # If last element is greater than sum, then ignore it 
    if arr[n-1] > _sum: 
        return isSubsetSum (arr, n-1, _sum) 
  
    ''' else, check if sum can be obtained by any of  
    the following 
    (a) including the last element 
    (b) excluding the last element'''
      
    return isSubsetSum (arr, n-1, _sum) or isSubsetSum (arr, n-1, _sum-arr[n-1]) 
  
    
        

#test case
arr = [3, 1, 5, 9, 12] 
print("Arr can be partitioned? ", partition(arr))    


#6##########################################################################################

'''
Subset Sum Problem


Given a set of non-negative integers, and a value sum, determine if there is a 
subset of the given set with sum equal to given sum.
Example:

Input: set[] = {3, 34, 4, 12, 5, 2}, sum = 9
Output: True 
    
Solution:
    recursion: time complexity exponential (NP complete: no known polynomial time sol)
        1 - Consider the last element and now the required sum = target 
        sum – value of ‘last’ element and number of elements = total 
        elements – 1
        2 - Leave the ‘last’ element and now the required sum = target sum and 
        number of elements = total elements – 1
        3 - Base Cases:
            isSubsetSum(set, n, sum) = false, if sum > 0 and n == 0
            isSubsetSum(set, n, sum) = true, if sum == 0 
    
    Dynamic Programming: time complexity O(n * targetSum)
        1 - we will create a 2D array of size (target + 1) x (arr.size() + 1) of 
        type boolean. The state DP[i][j] will be true if there exists a subset 
        of elements from A[0….i] with sum value = ‘j’. 

        if (A[i] > j)
        DP[i][j] = DP[i-1][j]
        else 
        DP[i][j] = DP[i-1][j] OR DP[i-1][sum-A[i]]
        
        This means that if current element has value greater than ‘current sum value’ 
        we will copy the answer for previous cases
        And if the current sum value is greater than the ‘ith’ element we will 
        see if any of previous states have already experienced the sum=’j’ OR 
        any previous states experienced a value ‘j – A[i]’ which will solve our 
        purpose.


set[]={3, 4, 5, 2}
target=6
 
    0    1    2    3    4    5    6

0   T    F    F    F    F    F    F

3   T    F    F    T    F    F    F
     
4   T    F    F    T    T    F    F   
      
5   T    F    F    T    T    T    F

2   T    F    T    T    T    T    T


'''
def subsetSum_rc(S, n, _sum):
     if _sum == 0:
         return True
     if n == 0:
         return False
     if _sum > S[n-1]:
         subsetSum_rc(S, n-1, _sum)
    
     return (subsetSum_rc(S, n-1, _sum) or subsetSum_rc(S, n-1, _sum - S[n-1]))
 
def subsetSum_dp(S, n, _sum):
    aux = [[False for i in range(_sum + 1)] for j in range(n + 1)]
    
    #if _sum == 0 return true
    for i in range(n + 1):
        aux[i][0] = True
    
    #if n == 0 return false
    for i in range(_sum + 1):
        aux[0][i] = False
        
    for i in range(1, n+1):
        for j in range(1, _sum+1):
            if j < S[i-1]:
                aux[i][j] = aux[i-1][j]
            else:
                aux[i][j] = aux[i-1][j] or aux[i-1][j-S[i-1]]
                
    return aux[n][_sum]


S = [3, 34, 4, 12, 5, 1] 
_sum = 9
n = len(S) 
print("S has a sum of _sum (recursion)? ", subsetSum_rc(S, n, _sum))
print("S has a sum of _sum (Dynamic Programming)? ", subsetSum_dp(S, n, _sum))


