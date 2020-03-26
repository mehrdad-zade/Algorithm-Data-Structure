'''
https://www.geeksforgeeks.org/index-mapping-or-trivial-hashing-with-negatives-allowed/

https://www.geeksforgeeks.org/find-whether-an-array-is-subset-of-another-array-set-1/

https://www.geeksforgeeks.org/find-duplicates-given-array-elements-not-limited-range/

https://www.geeksforgeeks.org/first-element-occurring-k-times-array/

https://www.geeksforgeeks.org/longest-subarray-sum-divisible-k/

https://www.geeksforgeeks.org/find-four-elements-a-b-c-and-d-in-an-array-such-that-ab-cd/
'''

def frequencyCount(array):
    #this can be done in java with hash
    map = dict()
    for e in array:
        if e in map.keys():
            map[e] += 1
        else:
            map[e] = 1
    
    print(map)
    
array = [1,3,4,2,7,5,3,2,5,1,1,7]
frequencyCount(array)