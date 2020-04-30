'''
https://www.geeksforgeeks.org/index-mapping-or-trivial-hashing-with-negatives-allowed/

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
