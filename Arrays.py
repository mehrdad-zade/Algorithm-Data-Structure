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
