
from sympy import *

NodeCount = 3

A = MatrixSymbol('A',NodeCount,NodeCount)
A = Matrix(A)

for i in xrange(NodeCount ** 2):
	A[i] = 'a' + repr(i)
	
print "Determinant of Square Matrix of rank "  + repr(NodeCount) 

print A.det()

print "Inverse:"

print simplify(A.inv() * A.det())
