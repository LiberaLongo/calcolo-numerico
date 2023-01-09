# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


print ("Hello world")
my_text = "Hello world"
a ="2"
print(a)


# # """ HELP """
# help(print) 



# # """Basic numerical types: int, float, complex"""


# # """Types and type conversions"""
a = 2
print(a, type(a))

a = float(a)
print(a, type(a))

""" NUMERI COMPLESSI """
c = 3.4+2.3j
print(c)
print(c.real)
print(c.imag)

# print ("%.5f" %c.imag)

print(int(4.255), float(7), complex(5.3))

my_text = "Hello world" 
print('my_text type:',type(my_text) ) # str is a string of character
print(len(my_text))



""" OPERAZIONI """
print(2+2)
print(2.0+2.0)
print(5/2)
print(5//2)
print(5%2)
print(3**2)



""" TYPE """
fl = 2/3
print(fl)
print (type(fl), "\n") 

""" PRINT """
print ("%f" %fl )
print ("%.2f" %fl) 
print ("%.20f" %fl)






# """Relational and boolean operators"""

x = 7
y = 7.0
z = 10

print (x==y)
print (x>z)
print (x<=y)
print (x!=y)

A = not (x==y) # NOT True
print ('A = ', A)
B = (x==y) and (x>z) # True AND False
print ('B = ', B)
C = (x==y) or (x>z) # True OR False
print('C = ', C)


# """Python modules"""
import math

x = 0 
print(math.sin(x))

from math import sqrt

x = 55.3
y = sqrt(x) 
print(y)

from math import pi
print(pi)

#help(math)
help(math.exp)


"""CICLI"""

"""IF"""
from random import random
from random import randint
x = random()
y = randint(1, 200)
print(x,'\n', y)

if y%2==0:
    print ('y is even')
else:
    print ('y is odd')
   


# """WHILE"""
print('y =', y)
k = 1
while k<=y: 
  print(k)
  k = k+1

print ('Last k', k)


"""FOR"""
primes = [2, 3, 5, 7]

print(primes)

for prime in primes:
    print(prime)

sequence = [i for i in range(10)]
print(sequence)



"""Tuples: immutable sequences of objects of any type"""
t = (1, 'hello', 3.14, True, (3.14, False)) # or () brackets!
print(t)
print(len(t))
print(t[0])
print(t[4])
print(t[-1])
print(type(t))
print(type(t[0]))
print(type(t[4]))

print(t[0:3])

print(3.14 in t )
print(t + (5, ))

print(t[0])
#t[0] = 2       #TypeError: 'tuple' object does not support item assignment

tt =  ('a',1)* 3 # concatenate
print(tt)

print(tt[0:len(tt):2])



"""Lists: mutable sequences of objects of any type"""

l = [1, 'hello', 3.14, True, (3.14, False) ] # square brackets
print(l)

l[0] = 2 # now it is allowed
print(l)

# as for tuples: 
print(len(l)) 
print(2.5 in l)
print(l[0:3])


v = []  # empty list
print(type(v), len(v))

v.append(5)
print(v)
v.append('hello')
print(v)


"""FUNCTIONS"""

def print_sum(x,y):
  """PRINT_SUM prints and returns the sum of two input numbers
  """
  s = x+y
  print('Sum of two input numbers: ', s)
  return s

a = 7
b = 2
my_sum = print_sum(a,b)
print("The returned value is ", my_sum)

s = print_sum( 3+5j , 7) # type compatibility
help(print_sum)



def sum_and_diff(x,y):
  """SUM_AND_DIFF computes sum and difference
  sum is the first output
  """
  return x+y, x-y # return as a tupla

help(sum_and_diff)

print(sum_and_diff(a,b))
S, D = sum_and_diff(a,b) # tupla
print(D)



# Variabili locali immutabili - interi

def myrandom1(x):
  x = random()
  return x

def myrandom2():
  a = random()
  return a

a = 2
print('myrandom1: ', myrandom1(a))
print('a = ', a)
print('myrandom2:', myrandom2())
print('a = ', a)


# Variabili locali mutabili - liste

v = [i for i in range(2, 9)]
print ('Original v: \t', v)
print(len(v))

def redouble(x):
  for i in range(0, len(x)):
    x[i] = 2*x[i]
  return x

print ('Output: \t', redouble(v))   # NOT v = redouble(v) 
print ('New v: \t\t', v)

def redouble2(x):
  d = []
  for i in range(0, len(x)):
    d.append(2*x[i])
  return d

v2 = redouble2(v)
print ('v input: \t', v)
print ('v2 output: \t', v2)







