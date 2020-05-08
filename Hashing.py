'''
https://www.geeksforgeeks.org/find-whether-an-array-is-subset-of-another-array-set-1/

https://www.geeksforgeeks.org/find-duplicates-given-array-elements-not-limited-range/

https://www.geeksforgeeks.org/first-element-occurring-k-times-array/

https://www.geeksforgeeks.org/longest-subarray-sum-divisible-k/

'''
#how many times has something been repeated in a list
def frequencyCount(array):
    #this can be done in java with hash
    map = dict()
    for e in array:
        if e in map.keys():
            map[e] += 1
        else:
            map[e] = 1

    print(map)

#Test Case
array = [1,3,4,2,7,5,3,2,5,1,1,7]
frequencyCount(array)

################################################################################
#find a, b, c, d in an array where a+b = c+d
def findPairs(arr):
  n = len(arr)
  sumOfPairs = dict()
  for i in range(0, n-1):
    for j in range(i+1, n):
      pairSum = arr[i] + arr[j]
      if pairSum in sumOfPairs.keys():
        return arr[i], arr[j], sumOfPairs.get(pairSum)[0], sumOfPairs.get(pairSum)[1]
      else:
        sumOfPairs[pairSum] = (arr[i], arr[j])
  return None, None, None, None

#Test Case
print(findPairs([3, 4, 7, 1, 2, 9, 8]))

################################################################################

'''
given two strings compare to see if there are common substrings between them

sol: create a hash array from all alphabets. go through the first string and mark the alphabet as true then
go through the second and compare
'''

def twoStrings(s1, s2) : 
    MAX_CHAR = 26
    v = [0] * (MAX_CHAR) 
      
    #ord gives the ascci number of a char
    for i in range(len(s1)): 
        v[ord(s1[i]) - ord('a')] = True
      
    # checking common substring  
    # of str2 in str1 
    for i in range(len(s2)) : 
        if (v[ord(s2[i]) - ord('a')]) : 
            return True
      
    return False

    
################################################################################

'''
Given a limited range array contains both positive and non-positive numbers, i.e., 
elements are in the range from -MAX to +MAX. Our task is to search if some number 
is present in the array or not in O(1) time.

sol: Since range is limited, we can use index mapping (or trivial hashing). We use 
values as the index in a big array. Therefore we can search and insert elements in O(1) time.

The idea is to use a 2D array of size hash[MAX+1][2]

Algorithm:

Assign all the values of the hash matrix as 0.
Traverse the given array:
    If the element ele is non negative assign 
        hash[ele][0] as 1.
    Else take the absolute value of ele and 
         assign hash[ele][1] as 1.
'''
maxValue = 1000

def isInRange(element, arr):
    hashMap = [
            [None]*(maxValue+1), #keeps positive values
            [None]*(maxValue)  #keeps ABS of negative values
        ]
    for e in arr:
        if e >=0 : 
            hashMap[0][e] = 1
        else:
            hashMap[1][abs(e)] = 1
    if element >= 0:
        if hashMap[0][element] == 1:
            return True
        else:
            return False
    else:
        if hashMap[1][abs(element)] == 1:
            return True
        else:
            return False
        
arr = [-1, 9, -5, -8, -5, -2] 
element = -5
print("Element is in arr? ", isInRange(element, arr))
        
################################################################################

'''
Find whether an array is subset of another array.

sol: sort both arrays and compare one by one. or, sort first arr and bin search elements of second arr
'''

def isSubset(arr1, arr2):
    arr1.sort()
    arr2.sort()
    
    i = 0
    j = 0
    
    n = len(arr1)
    m = len(arr2)
    
    if n < m:
        return False
    
    print(arr1)
    print(arr2)
    
    while i < n and j < m:
        if arr1[i] == arr2[j]:
            i += 1
            j += 1
        elif arr1[i] > arr2[j]:
            return False
        else:
            i += 1
    if j < m:
        return False
    return True
            
                
   

arr1 = [11, 1, 13, 21, 3, 7]; 
arr2 = [11, 1, 13, 21, 3]; 
print("is arr2 a subset of arr1? ", isSubset(arr1, arr2))

################################################################################
