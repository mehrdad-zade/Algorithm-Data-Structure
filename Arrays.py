#1###############################################################################

'''
this problem gives you an array with all zeros. you have to go through 
the queries which provides you with a range and val. the
val must be added to all the elements in that range. return the maximum 
alue in the resulting arr.
sol: add the value to the begining of the range and substract val from 
he end of the range + 1. then loop through the array and
do a prefix addition (this will add the value for each of the elements 
ithin expected range. +1 for end of range is because
you only want to add values to elements in the range inclusively, so by 
ubtracting from end+1 you will start the next range with
prefix addition without considering the value from previous range)
'''
def maxRangeAdditions(n, queries):
    arr = [0 for i in range(n)]
    queries_Rows = 0
    while queries_Rows < len(queries) :
        arr[queries[queries_Rows][0] - 1] += queries[queries_Rows][2]
        if queries[queries_Rows][1] != n :
            arr[queries[queries_Rows][1]] -= queries[queries_Rows][2]
        queries_Rows += 1

    maxVal = arr[0]
    for i in range(1,n,1) :
        arr[i] += arr[i-1]
        if arr[i] > maxVal :
            maxVal = arr[i]
    return maxVal

n = 9
queries = [[1, 5, 3],
           [4, 8, 7],
           [6, 9, 1]]
print("array addition, max occurence with added num: ", maxRangeAdditions(n, queries))

#2#######################################################################################################################

'''
find all the subarrays
'''

def subArrays(arr):
    sub = [[]]
    for i in range(len(arr)+1):
        for j in range(i, len(arr)+1):
            temp = arr[i:j]
            if temp:
                sub.append(temp)
    return sub

arr = [1, 2, 3] 
print("sub-arrays = ", subArrays(arr)) 

#3###############################################################################

'''
Find whether an array is subset of another array.

sol: sort both arrays and compare one by one. or, sort first arr and bin search elements of second arr
'''

def isSubArr(arr1, arr2):#find out if arr2 is a sub-array of arr1
    n1 = len(arr1)
    n2 = len(arr2)
    if n2 > n1: # if arr2 is bigger, then it cannot be the sub-array of arr1
        return False
    arr1.sort()
    arr2.sort()
    j = 0
    for i in range(n1):
        print("i = ", i)
        print("j = ", j)
        if arr1[i] == arr2[j]:
            j += 1
        elif arr1[i] > arr2[j]:
            return False
        else: # if arr1<arr2
            continue
    if j != n2:
        return False
    return True

arr1 = [11, 1, 13, 21, 3, 7]; 
arr2 = [11, 1, 13, 21, 7]; 
print("is arr2 a subset of arr1? ", isSubArr(arr1, arr2))

#4###############################################################################
'''
rotate elements of the array d times. if d=2 and array is 1,2,3,4; then one time rotation would result
in: arry = 3,4,1,2.
suppose d is static and you will apply it many times. write an efficient program to rotate
the array eleemnts.

Solution: you need to find the length of the array, n. and then find the GCD of n and d, gcd.
you will create gcd number of sets. each set will contain values that are d elements appart.
i.e. if array = 1,2,3,4,5,6,7,8,910,11,12; d=3; gcd=3; then
set1 = {1,4,7,10}
set2 = {2,5,8,11}
set3 = {3,6,9,12}
now each time you have to rotate the array by d=3 number of elements, you only have to swap
the first and last elements of each set:
set1 = {10,4,7,1}
set2 = {11,5,8,2}
set3 = {12,6,9,3}

#5###############################################################################
'''
Job sequencing O(n^2)

greedy algorithm

we get a list of jobs with IDs, deadline (in unit of time), and profit.

JobID  Deadline  Profit
  a      4        20   
  b      1        10
  c      1        40  
  d      1        30
  
we want to maximize profit if only one job can be scheduled at a time.

solution is to sort the table based on profit. 
take the highest value and check the deadline. if it's 3 for instance, you have to put the ID
in an array on the third slot. if third slot is full, go one back until you cannot go further back.
if there is no place to put it then we should skip the job  
'''

def jobSequencing(jobs):
    n = len(jobs)
    temp_sequence = [0 for i in range(n)]
    jobs_sorted = sortJobsOnProfit(jobs)

    for i in range(n-1, -1, -1):

        j = jobs_sorted[i][1] - 1
      
        while j >= 0 :
            if temp_sequence[j] == 0:
                temp_sequence[j] = jobs_sorted[i][0]
                break
            j -= 1
    sequence = []
    for e in temp_sequence :
        if e != 0:
            sequence.append(e)
    return sequence
    
def sortJobsOnProfit(jobs):
    return sorted(jobs, key = lambda j : j[2])
    
    
jobs = [['a',4,20], 
        ['b', 1, 10],
        ['c', 1, 40],
        ['d', 1, 30]
        ]    

print("Job Sequence to Maximize Profit : ", jobSequencing(jobs))


##############################################################################
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

##############################################################################
'''
Given a value N, if we want to make change for N cents, and we have infinite 
supply of each of S = { S1, S2, .. , Sm} valued coins, how many ways can we 
make the change? The order of coins doesn’t matter.

For example, for N = 4 and S = {1,2,3}, there are four solutions: {1,1,1,1},
{1,1,2},{2,2},{1,3}. So output should be 4. For N = 10 and S = {2, 5, 3, 6}, 
there are five solutions: {2,2,2,2,2}, {2,2,3,3}, {2,2,6}, {2,3,5} and {5,5}. 
So the output should be 5.

yu can see the recursive solution for better understanding the way to solve this, but
we can do it with dynamic programming to save time O(mn)

'''

# Returns the count of ways we can sum 
# S[0...m-1] coins to get sum n 
def count1(S, m, n ): 
  
    # If n is 0 then there is 1 
    # solution (do not include any coin) 
    if (n == 0): 
        return 1
  
    # If n is less than 0 then no 
    # solution exists 
    if (n < 0): 
        return 0; 
  
    # If there are no coins and n 
    # is greater than 0, then no 
    # solution exist 
    if (m <=0 and n >= 1): 
        return 0
  
    # count is sum of solutions (i)  
    # including S[m-1] (ii) excluding S[m-1] 
    return count( S, m - 1, n ) + count( S, m, n-S[m-1] );

def count2(S, m, n):
    table = [0 for i in range(n+1)]
    table[0] = 1
    for i in range(m):
        for j in range(S[i], n+1):
            table[j] += table[j-S[i]]
    return table
    
    
arr = [1, 2, 3] 
m = len(arr) 
n = 4
print(count2(arr, m, n))
#########################################################################################
'''
Matrix Chain Multiplication | DP-8

Last Updated: 25-08-2020
Given a sequence of matrices, find the most efficient way to multiply these matrices together. The problem is not actually to perform the multiplications, but merely to decide in which order to perform the multiplications.

We have many options to multiply a chain of matrices because matrix multiplication is associative. In other words, no matter how we parenthesize the product, the result will be the same. For example, if we had four matrices A, B, C, and D, we would have:

    (ABC)D = (AB)(CD) = A(BCD) = ....
However, the order in which we parenthesize the product affects the number of simple arithmetic operations needed to compute the product, or the efficiency. For example, suppose A is a 10 × 30 matrix, B is a 30 × 5 matrix, and C is a 5 × 60 matrix. Then,

    (AB)C = (10×30×5) + (10×5×60) = 1500 + 3000 = 4500 operations
    A(BC) = (30×5×60) + (10×30×60) = 9000 + 18000 = 27000 operations.


 Input: p[] = {40, 20, 30, 10, 30}   
  Output: 26000  
  There are 4 matrices of dimensions 40x20, 20x30, 30x10 and 10x30.
  Let the input 4 matrices be A, B, C and D.  The minimum number of 
  multiplications are obtained by putting parenthesis in following way
  (A(BC))D --> 20*30*10 + 40*20*10 + 40*10*30    
'''

# A naive recursive implementation that 
# simply follows the above optimal  
# substructure property. there is a complex dynamic solution to it too
import sys 
  
# Matrix A[i] has dimension p[i-1] x p[i] 
# for i = 1..n 
def MatrixChainOrder(p, i, j): 
  
    if i == j: 
        return 0
  
    _min = sys.maxsize 
      
    # place parenthesis at different places  
    # between first and last matrix,  
    # recursively calculate count of 
    # multiplications for each parenthesis 
    # placement and return the minimum count 
    for k in range(i, j): 
      
        count = (MatrixChainOrder(p, i, k)  
             + MatrixChainOrder(p, k + 1, j) 
                   + p[i-1] * p[k] * p[j]) 
  
        if count < _min: 
            _min = count; 
      
  
    # Return minimum count 
    return _min; 
  
  
# Driver program to test above function 
arr = [1, 2, 3, 4, 3]; 
n = len(arr); 
  
print("Minimum number of multiplications is ", 
                MatrixChainOrder(arr, 1, n-1)); 

#########################################################################################
                