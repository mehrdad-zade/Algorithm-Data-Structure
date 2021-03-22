#1###################################################################################################################

'''
Median of two sorted arrays of same size

There are 2 sorted arrays A and B of size n each. Write an algorithm to find 
the median of the array obtained after merging the above 2 arrays(i.e. array of 
length 2n). The complexity should be O(log(n)). 

1) Calculate the medians m1 and m2 of the input arrays ar1[] 
   and ar2[] respectively.
2) If m1 and m2 both are equal then we are done.
     return m1 (or m2)
3) If m1 is greater than m2, then median is present in one 
   of the below two subarrays.
    a)  From first element of ar1 to m1 (ar1[0...|_n/2_|])
    b)  From m2 to last element of ar2  (ar2[|_n/2_|...n-1])
4) If m2 is greater than m1, then median is present in one    
   of the below two subarrays.
   a)  From m1 to last element of ar1  (ar1[|_n/2_|...n-1])
   b)  From first element of ar2 to m2 (ar2[0...|_n/2_|])
5) Repeat the above process until size of both the subarrays 
   becomes 2.
6) If size of the two arrays is 2 then use below formula to get 
  the median.
    Median = (max(ar1[0], ar2[0]) + min(ar1[1], ar2[1]))/2
Examples :  

   ar1[] = {1, 12, 15, 26, 38}
   ar2[] = {2, 13, 17, 30, 45}
For above two arrays m1 = 15 and m2 = 17
For the above ar1[] and ar2[], m1 is smaller than m2. So median is present in one of the following two subarrays. 
 

   [15, 26, 38] and [2, 13, 17]
Let us repeat the process for above two subarrays: 
 

    m1 = 26 m2 = 13.
m1 is greater than m2. So the subarrays become  

  [15, 26] and [13, 17]
Now size is 2, so median = (max(ar1[0], ar2[0]) + min(ar1[1], ar2[1]))/2
                       = (max(15, 13) + min(26, 17))/2 
                       = (15 + 17)/2
                       = 16
'''

def getMedian(arr1, arr2, n): 
     
    # there is no element in any array
    if n == 0: 
        return -1
         
    # 1 element in each => median of 
    # sorted arr made of two arrays will    
    elif n == 1: 
        # be sum of both elements by 2
        return (arr1[0]+arr2[1])/2
         
    # Eg. [1,4] , [6,10] => [1, 4, 6, 10]
    # median = (6+4)/2    
    elif n == 2: 
        # which implies median = (max(arr1[0],
        # arr2[0])+min(arr1[1],arr2[1]))/2
        return (max(arr1[0], arr2[0]) +
                min(arr1[1], arr2[1])) / 2
     
    else:
        #calculating medians     
        m1 = median(arr1, n)
        m2 = median(arr2, n)
         
        # then the elements at median 
        # position must be between the 
        # greater median and the first 
        # element of respective array and 
        # between the other median and 
        # the last element in its respective array.
        if m1 > m2:
             
            if n % 2 == 0:
                return getMedian(arr1[:int(n / 2) + 1],
                        arr2[int(n / 2) - 1:], int(n / 2) + 1)
            else:
                return getMedian(arr1[:int(n / 2) + 1], 
                        arr2[int(n / 2):], int(n / 2) + 1)
         
        else:
            if n % 2 == 0:
                return getMedian(arr1[int(n / 2 - 1):],
                        arr2[:int(n / 2 + 1)], int(n / 2) + 1)
            else:
                return getMedian(arr1[int(n / 2):], 
                        arr2[0:int(n / 2) + 1], int(n / 2) + 1)
 
 # function to find median of array
def median(arr, n):
    if n % 2 == 0:
        return (arr[int(n / 2)] +
                arr[int(n / 2) - 1]) / 2
    else:
        return arr[int(n/2)]
 
     
# Driver code
arr1 = [1, 2, 3, 6]
arr2 = [4, 6, 8, 10]
n = len(arr1)
print(int(getMedian(arr1,arr2,n)))

####################################################################################################################
'''
Closest Pair of Points : O(nLogn)

We are given an array of n points in the plane, and the problem is to find out the closest pair of points in the array. 
'''


####################################################################################################################
