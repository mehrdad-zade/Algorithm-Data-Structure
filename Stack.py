#https://www.geeksforgeeks.org/stack-set-2-infix-to-postfix/

################################################################################
def HanoiTower(n, Tower1, Tower2, Tower3):
  if n==1:
    print ("Move disk",n,"from tower",Tower1,"to tower",Tower2)
    return
  HanoiTower(n-1, Tower1, Tower3, Tower2)
  print ("Move disk",n,"from tower",Tower1,"to tower",Tower2)
  HanoiTower(n-1, Tower3, Tower2, Tower1)


#Test Case
n=3
HanoiTower(n, 'A', 'B', 'C')

################################################################################
# find the next greater element in an array. if there are none print -1
#nextGreaterElement1 : time complexity is O(n^2)
def nextGreaterElement1(arr):
    nextGreaterArr = []
    n = len(arr)
    for i in range(n):
        for j in range(i,n):
            if arr[j] > arr[i]:
                nextGreaterArr.append(arr[j])
                break
            if j==n-1:
                nextGreaterArr.append(-1)
    return nextGreaterArr

#nextGreaterElement2 : using stack time complexity it O(n)
def nextGreaterElement2(arr):
    stack = []
    stackElement = 0
    arrElement = 0

    stack.append(arr[0])
    for i in range(1,len(arr)):
        arrElement = arr[i]
        if len(stack) != 0:
            stackElement = stack.pop()
            while stackElement < arrElement:
                print(str(stackElement) + " --> " + str(arrElement))
                if len(stack) == 0:
                    break
                stackElement = stack.pop()
            if stackElement > arrElement:
                stack.append(stackElement)
        stack.append(arrElement)
    while len(stack) != 0:
        stackElement = stack.pop()
        arrElement = -1
        print(str(stackElement) + " --> " + str(arrElement))

#Test Case
arr = [4, 5, 2, 25, 20]
#arr = [11, 13, 21, 3]
print(nextGreaterElement1(arr))
print(nextGreaterElement2(arr))

################################################################################
'''
Given an expression string exp , write a program to examine whether the pairs
and the orders of “{“,”}”,”(“,”)”,”[“,”]” are correct in exp.
Input: exp = “[()]{}{[()()]()}”
Output: Balanced

Input: exp = “[(])”
Output: Not Balanced
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 22:42:03 2020

@author: mehrdadalemzadeh
"""

def isBalanced(exp):
    stack = []
    for e in exp:
        if e in ["[", "{", "("]:
            stack.append(e)
        elif e in ["]", "}", ")"]:
            temp = stack.pop()
            if (temp == "(" and e != ")") or (temp == "{" and e != "}") or (temp == "[" and e != "]"):
                return False
    if len(stack) == 0:
        return True
    else:
        return

#Test Case
exp1 = "[()]{}{[()()]()}"
exp2 = "[(])"
print("Expression 1 is balanced? ", isBalanced(exp1))
print("Expression 2 is balanced? ", isBalanced(exp2))

################################################################################
