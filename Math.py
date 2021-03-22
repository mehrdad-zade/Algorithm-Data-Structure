#1##########################################################################################

'''
Find day of the week for a given date

https://www.youtube.com/watch?v=dD8iXe_InnQ
'''
def dayofweek(d, m, y): 
    t = [ 0, 3, 2, 5, 0, 3, 
          5, 1, 4, 6, 2, 4 ] 
    y -= m < 3
    return (( y + int(y / 4) - int(y / 100) 
             + int(y / 400) + t[m - 1] + d) % 7) 
  
# Driver Code 
day = dayofweek(30, 8, 2010) 
print(day) 

#2#################################################################

'''
How to check if a given number is Fibonacci number?

A simple way is to generate Fibonacci numbers until the generated number is greater than or equal to ‘n’. 
Following is an interesting property about Fibonacci numbers that can also be used to check if a given 
number is Fibonacci or not. A number is Fibonacci if and only if one or both of (5*n^2 + 4) or (5*n^2 – 4) 
is a perfect square
'''

import math 
  
# A utility function that returns true if x is perfect square 
def isPerfectSquare(x): 
    s = int(math.sqrt(x)) 
    return s*s == x 
  
# Returns true if n is a Fibinacci Number, else false 
def isFibonacci(n): 
  
    # n is Fibinacci if one of 5*n*n + 4 or 5*n*n - 4 or both 
    # is a perferct square 
    return isPerfectSquare(5*n*n + 4) or isPerfectSquare(5*n*n - 4) 
     
# A utility function to test above functions 
for i in range(1,11): 
     if (isFibonacci(i) == True): 
         print i,"is a Fibonacci Number"
     else: 
         print i,"is a not Fibonacci Number "

#3#################################################################
'''
Program for Tower of Hanoi


'''         

def TowerOfHanoi(n , from_rod, to_rod, aux_rod):
    if n == 1:
        print("Move disk 1 from rod",from_rod,"to rod",to_rod)
        return
    TowerOfHanoi(n-1, from_rod, aux_rod, to_rod)
    print("Move disk",n,"from rod",from_rod,"to rod",to_rod)
    TowerOfHanoi(n-1, aux_rod, to_rod, from_rod)
         
# Driver code
n = 4
TowerOfHanoi(n, 'A', 'C', 'B') 
# A, C, B are the name of rods

#4#################################################################
'''
Add two numbers without using arithmetic operators

Sum of two bits can be obtained by performing XOR (^) of the two bits. Carry bit can be obtained by performing AND (&) of two bits. 
Above is simple Half Adder logic that can be used to add 2 single bits. We can extend this logic for integers. If x and y don’t 
have set bits at same position(s), then bitwise XOR (^) of x and y gives the sum of x and y. To incorporate common set bits also, 
bitwise AND (&) is used. Bitwise AND of x and y gives all carry bits. We calculate (x & y) << 1 and add it to x ^ y to get the required result. 
'''
# without using arithmetic operator
def Add(x, y):
 
    # Iterate till there is no carry 
    while (y != 0):
     
        # carry now contains common
        # set bits of x and y
        carry = x & y
 
        # Sum of bits of x and y where at
        # least one of the bits is not set
        x = x ^ y
 
        # Carry is shifted by one so that   
        # adding it to x gives the required sum
        y = carry << 1
     
    return x
 
print(Add(15, 32))


#5#################################################################

'''
Add 1 to a given number

Write a program to add one to a given number. The use of operators like 
‘+’, ‘-‘, ‘*’, ‘/’, ‘++’, ‘–‘ …etc are not allowed.

sol.
To add 1 to a number x (say 0011000111), flip all the bits after the 
rightmost 0 bit (we get 0011000000). Finally, flip the rightmost 0 bit 
also (we get 0011001000) to get the answer.
'''
# one to a given number  
def addOne(x) : 
      
    m = 1; 
    # Flip all the set bits 
    # until we find a 0  
    while(x & m): 
        x = x ^ m 
        m <<= 1
      
    # flip the rightmost  
    # 0 bit  
    x = x ^ m 
    return x 
  
# Driver program 
n = 13
print addOne(n) 
  
#6#################################################################
'''
Check for Integer Overflow

There can be overflow only if signs of two numbers are same, and sign of sum is opposite to the signs of numbers.

1)  Calculate sum
2)  If both numbers are positive and sum is negative then return -1
     Else 
        If both numbers are negative and sum is positive then return -1
        Else return 0
'''
#7#################################################################

'''
Binary representation of a given number

For any number, we can check whether its ‘i’th bit is 0(OFF) or 1(ON) by bitwise ANDing it with “2^i” (2 raise to i). 

1) Let us take number 'NUM' and we want to check whether it's 0th bit is ON or OFF    
    bit = 2 ^ 0 (0th bit)
    if  NUM & bit == 1 means 0th bit is ON else 0th bit is OFF

2) Similarly if we want to check whether 5th bit is ON or OFF    
    bit = 2 ^ 5 (5th bit)
    if NUM & bit == 1 means its 5th bit is ON else 5th bit is OFF.
'''

def bin(n) :
     
    i = 1 << 31
    while(i > 0) :
     
        if((n & i) != 0) :
         
            print("1", end = "")
         
        else :
            print("0", end = "")
             
        i = i // 2
             
bin(7)
print()
bin(4)
 
 
