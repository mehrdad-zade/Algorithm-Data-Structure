#1##########################################################################################

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

#2###############################################################################
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

#3###############################################################################
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

#4###############################################################################

'''
Infix expression:The expression of the form a op b. When an operator is
in-between every pair of operands.

Postfix expression:The expression of the form a b op. When an operator is
followed for every pair of operands.

benifit: compiler has to re-assess the expression for infix to observe op
precidance, but with postfix(or prefix) it will be a left to right assessment
'''

def infix2Postfix(exp):
    precedanceMap = {"^": 3, "*":2, "/":2, "+":1, "-":1}
    stack = []
    valuePosition = 0 #keeps the position of the operands to be added to postfixArr
    postfixArr = []
    for i in range(len(exp)):
        if exp[i] in precedanceMap.keys() or exp[i] in ["(", ")"]: #we want to evaluate only op's
            #first add the value based on the last position. positions are set when op is met. omit pranthesys and ""
            if exp[valuePosition:i] not in ["(", ")"] and exp[valuePosition:i] != "":
                postfixArr.append(exp[valuePosition:i])
            valuePosition = i+1#move the position forward even if the element met was "(" or ")"
            #if stack is empty or exp is "(" or top of stack has "(" add the op to stack
            if len(stack) == 0 or exp[i] == "(" or stack[len(stack)-1] == "(":
                stack.append(exp[i])
            #added seperate elif to handle "(" and ")" in precedance check. if precedance of new op is lower than top of the stack
            elif exp[i] not in ["(", ")"] and precedanceMap[stack[len(stack)-1]] <= precedanceMap[exp[i]]:
                stack.append(exp[i])
            #if met ")" pop op's until you meet "("
            elif exp[i] == ")":
                    while len(stack) > 0:
                        if stack[-1] != "(":
                            postfixArr.append(stack.pop())
                        else:
                            stack.pop()#once "(" was met stop poping
                            break
            #if precedance of stack is higher, first pop all stack then add newly met op
            else:
                while len(stack) > 0:
                    if stack[-1] != "(":
                        postfixArr.append(stack.pop())
                    else:
                        break #if you saw an open pranthesis stop poping from stack
                stack.append(exp[i])#adding newly met
    #add the last value and stack
    if exp[valuePosition:len(exp)] != "":
        postfixArr.append(exp[valuePosition:len(exp)])
    while len(stack)>0:
        postfixArr.append(stack.pop())

    return postfixArr





exp1 = "a+b*c+d"
exp2 = "a+b*(c^d-e)^(f+g*h)-i" # abcd^e-fgh*+^*+i-
print(infix2Postfix(exp1))
print(infix2Postfix(exp2))


