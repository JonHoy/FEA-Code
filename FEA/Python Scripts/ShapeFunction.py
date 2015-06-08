from sympy import *
from sympy.integrals.quadrature import gauss_legendre
import numpy as np
import numpy.linalg as linalg
import math

Order = 2
CarVals = ['x','y','z']
Vals = ['r' , 's' , 't']
Coeffs = ['a','b','c']
Range = np.linspace(-1,1,Order)

		
Dims = len(Vals)
N = Order ** Dims # number of coeffs
	  

Size = []

for i in xrange(Dims):
	Size.append(Order)
RangePts = np.zeros((Order ** len(Vals),len(Vals)))

CoeffMatrix = np.ones((RangePts.shape[0],RangePts.shape[0]),dtype = np.double)

for i in xrange(RangePts.shape[0]):
	Sub = np.unravel_index(i,Size)
	for j in xrange(len(Size)):
		RangePts[i,j] = Range[Sub[j]];

for k in xrange(len(Coeffs)):
	Poly = ""
	for i in xrange(N):
		if i != 0:
			Poly = Poly + " + " + Coeffs[k] + repr(i) + "*" 
		Sub = np.unravel_index(i,Size)
		for j in xrange(Dims):
			if j != 0:
				Poly = Poly + "*"	
			Poly = Poly + Vals[j] + "**" + repr(Sub[j])
	exec(CarVals[k] + " = Poly")


for i in xrange(CoeffMatrix.shape[0]):
	for j in xrange(CoeffMatrix.shape[1]):
		Sub = np.unravel_index(j,Size)	
		for k in xrange(RangePts.shape[1]):
			CoeffMatrix[i,j] = CoeffMatrix[i,j] * RangePts[i,k] ** Sub[k]

CoeffMatrix = np.matrix(CoeffMatrix)
print linalg.inv(CoeffMatrix)

Jacobian = MatrixSymbol('J',len(CarVals),len(Vals))
Jacobian = Matrix(Jacobian)
for i in xrange(len(CarVals)):
		for j in xrange(len(Vals)):
			exec(("Jacobian[i,j] = diff(" + CarVals[i] + ",'" + Vals[j]  + "')"))	
#print Jacobian	

GOrder = int(math.ceil((Order + 1) / 2.0))


# now that we have the jacobian, we can use gauss quadrature to evaluate the determinant at discrete points to get the exact answer for up polynomials of order 2n-1, so if we have an element of order n we must have quadrature of at lease (n + 1) / 2 

Locations, Weights = gauss_legendre(GOrder, 15)

GSize = []
for i in xrange(Dims):
	GSize.append(GOrder)
EvalPts = np.zeros((len(Locations) ** len(Vals),len(Vals)))
EvalWeights = np.ones(EvalPts.shape[0])

for i in xrange(EvalPts.shape[0]):
	Sub = np.unravel_index(i,GSize)
	for j in xrange(len(Vals)):
		EvalPts[i,j] = Locations[Sub[j]]
		EvalWeights[i] = EvalWeights[i] * Weights[Sub[j]]

print Jacobian

JacobianList = []
#for k in xrange(len(EvalWeights)):
#	myJacobian = Jacobian.copy()
#	for l in xrange(len(Vals)):
#		myJacobian = myJacobian.subs(Vals[l], EvalPts[k,l])
#	JacobianList.append(myJacobian)

