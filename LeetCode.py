'''
Given an integer array nums, you need to find one continuous subarray that if you only sort this 
subarray in ascending order, then the whole array will be sorted in ascending order.

Return the shortest such subarray and output its length.

 

Example 1:

Input: nums = [2,6,4,8,10,9,15]
Output: 5
Explanation: You need to sort [6, 4, 8, 10, 9] in ascending order to make the whole array sorted in ascending order.
'''
class Solution:
    def findUnsortedSubarray(self, N: List[int]) -> int:
        lenN, left, right = len(N) - 1, -1, -1
        maxN, minN = N[0], N[lenN]
        for i in range(1, len(N)):
            a, b = N[i], N[lenN-i]
            if a < maxN: right = i
            else: maxN = a
            if b > minN: left = i
            else: minN = b
        return max(0, left + right - lenN + 1)

##############################################################################################################################
'''
Given two integers dividend and divisor, divide two integers without using multiplication, division, and mod operator.

Return the quotient after dividing dividend by divisor.

Example 1:

Input: dividend = 10, divisor = 3
Output: 3
Explanation: 10/3 = truncate(3.33333..) = 3.
'''

class Solution:
    def divide(self, dividend: int, divisor: int) -> int:    
        
        sign = -1 if ((dividend < 0) ^  (divisor < 0)) else 1 # find the result's sign
        quotient = 0
        
        dividend = abs(dividend)
        divisor = abs(divisor)
        
        while (dividend >= divisor):
            quotient += 1
            dividend -= divisor
            
        return quotient * sign


class Solution2: # https://www.youtube.com/watch?v=htX69j1jf5U
    def divide(self, dividend: int, divisor: int) -> int:    
        
        sign = (-1 if((dividend < 0) ^ 
                  (divisor < 0)) else 1);
     
        # remove sign of operands
        dividend = abs(dividend);
        divisor = abs(divisor);

        # Initialize
        # the quotient
        quotient = 0;
        temp = 0;

        # test down from the highest 
        # bit and accumulate the 
        # tentative value for valid bit
        for i in range(31, -1, -1):
            if (temp + (divisor << i) <= dividend):
                temp += divisor << i;
                quotient |= 1 << i;

        return sign * quotient;




##############################################################################################################################

'''
Check if two strings can be made equal by swapping one character among each other
Difficulty Level : Basic
 Last Updated : 28 Nov, 2019
Given two strings A and B of length N, the task is to check whether the two strings can be made equal by swapping any character 
of A with any other character of B only once.

Examples:

Input: A = “SEEKSFORGEEKS”, B = “GEEKSFORGEEKG”
Output: Yes
“SEEKSFORGEEKS” and “GEEKSFORGEEKG”
can be swapped to make both the strings equal.

Input: A = “GEEKSFORGEEKS”, B = “THESUPERBSITE”
Output: No

Recommended: Please try your approach on {IDE} first, before moving on to the solution.
Approach: First omit the elements which are the same and have the same index in both the strings. Then if the new strings are of 
length two and both the elements in each string are the same then only the swap is possible.


'''



##############################################################################################################################

'''
Given a string s consisting only of letters 'a' and 'b'. In a single step you can remove one palindromic subsequence from s.

Return the minimum number of steps to make the given string empty.

A string is a subsequence of a given string, if it is generated by deleting some characters of a given string without changing its order.

A string is called palindrome if is one that reads the same backward as well as forward.

sol.
result is 0, 1 or 2. if s is empty then 0. if there exisits a palindrome then 1, else 2 (We can remove any of the two subsequences to 
get a unary string. A unary string is always palindrome.)
'''
class Solution:
    def removePalindromeSub(self, s: str) -> int:
        if s == '':
            return 0
        elif self.isPalindrome(s):
            return 1
        else:
            return 2
        
    def isPalindrome(self, s):
        n = len(s)
        i, j = 0, n-1
        while i <= j:
            if s[i] != s[j]:
                return False
            i += 1
            j -= 1
        return True

##############################################################################################################################
'''
You are given coins of different denominations and a total amount of money amount. Write a function to compute the fewest 
number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.

 

Example 1:

Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1
'''

Naive greedy won't work.
Consider the case where coins = [3,7], amount = 16.
Keep track of dp[c][i] meaning the smallest number of coins to make amount i using the first c coins.
dp[0][0] = 0 and dp[0][i] = inf for i > 0.
dp[c+1][i] is the smaller of dp[c][i] -- not making use of the (c+1)-th coin; or dp[c+1][i-coins[c]] + 1 -- add one more c-th coin to the best solution that makes up an amount of i-coins[c] using only the first c coins.
No need to actually maintain space for a 2-D array if we iterate along c.
Code

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [0] + [inf] * amount
        for x in coins:
            for i in range(amount - x + 1):
                dp[i + x] = min(dp[i + x], dp[i] + 1)
        return dp[amount] if dp[amount] != inf else -1
##############################################################################################################################

'''
Given a binary string s and an integer k.

Return True if every binary code of length k is a substring of s. Otherwise, return False.

 

Example 1:

Input: s = "00110110", k = 2
Output: true
Explanation: The binary codes of length 2 are "00", "01", "10" and "11". They can be all found as substrings at indicies 0, 1, 3 and 2 respectively.
'''

class Solution:
    def hasAllCodes(self, s: str, k: int) -> bool:
        us = set()
        for i in range(len(s) -k + 1) :    
            us.add(s[i : k+i])    
        return len(us) == 1 << k



########################################################################################################################

'''
Given an array of unique integers, arr, where each integer arr[i] is strictly greater than 1.

We make a binary tree using these integers, and each number may be used for any number of times. 
Each non-leaf node's value should be equal to the product of the values of its children.

Return the number of binary trees we can make. The answer may be too large so return the answer modulo 109 + 7.


dea:

The trick to this problem is realizing that we can break it down into smaller pieces. A number can always be a leaf, 
so the number of ways it can form a branch should always start at 1.

If the number can be made from multiple factor pairs, then ways is our starting value of 1 plus the sum of all the ways 
to make those factor pairs.

For each existing factor pair (fA & fB), the number of ways to make that that particular pair configuration is the product 
of the number of ways to make fA and fB.

So we can see that each number relies on first solving the same question for each of its factors. This means that we should 
start by sorting our numbers array (A). Then we can iterate through A and figure out each number in ascending order, so that 
we will have completed any factors for larger numbers before we need to use them.

This means storing the information, which we can do in a map, so that we can look up the results by value.

In order to be more efficient when we attempt to find each factor pair, we only need to iterate through A up to the square 
root of the number in question, so that we don't duplicate the same factor pairs going the opposite direction. That means 
we need to double every pair result where fA and fB are not the same.

Since each number can be the head of a tree, our answer (ans) will be the sum of each number's result. We shouldn't forget 
to modulo at each round of summation.
'''

class Solution:
    def numFactoredBinaryTrees(self, A: List[int]) -> int:
        A.sort()
        fmap, ans = defaultdict(), 0
        for num in A:
            ways, lim = 1, sqrt(num)
            for fA in A:
                if fA > lim: break
                fB = num / fA
                if fB in fmap:
                    ways += fmap[fA] * fmap[fB] * (1 if fA == fB else 2)
            fmap[num], ans = ways, (ans + ways)
        return ans % 1000000007
