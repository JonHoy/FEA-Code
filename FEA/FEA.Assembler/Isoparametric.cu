
#include "Polynomial.cuh"
#include "Inverse.cuh"
#include "Determinant.cuh"
#include "CoefficientMatrix.cuh"
#include "GaussQuadrature.cuh"


// put unique specifiers up here
#define Size 8
#define Dims 3
typedef float T;
#define Order 2
#define QuadOrder 2


extern "C" __global__ void JacobianDeterminant(
const int ElementCount, // how many elements are on gpu   
int* NodeIds,
T* X, // x coordinates of each of the nodes
T* Y, // y coordinates of each of the nodes
T* Z, // z coordinates of each of the nodes
T* Determinant) { // determinants of the jacobian at locations specified by gauss quadrature  

// X is a Matrix of (Size x ElementCount

CoefficientMatrix<T,Size,Dims> C; // matrix that when multiplied by xyz coordinates maps to rst

T xU[Size];
T yU[Size];
T zU[Size];

Polynomial<T,Order,Order,Order> x; // polynomial mapping x to rst
Polynomial<T,Order,Order,Order> y; // map y to rst
Polynomial<T,Order,Order,Order> z; // map z to rst

int iD = blockIdx.x * blockDim.x + threadIdx.x;

for (int i = 0; i < Size; i++) {
	int Location = NodeIds[iD * Size + i];
	xU[i] = X[Location];
	yU[i] = Y[Location];
	zU[i] = Z[Location];
}		

C.GetAlpha( x.Coeffs, xU);
C.GetAlpha( y.Coeffs, yU);
C.GetAlpha( z.Coeffs, zU);

// once we get the coeffients we must make the jacobian
Quadrature<float, QuadOrder> GaussPts;

T Jacobian[3][3];

int counter;
for (int i = 0; i < QuadOrder; i++)
{
	T ValR = GaussPts.Weights[i];
	for (int j = 0; j < QuadOrder; j++)
	{
		T ValS = GaussPts.Weights[j];
		for (int k = 0; k < QuadOrder; k++)
		{
			T ValT = GaussPts.Weights[k];
			Jacobian[0][0] = x.Differentiate(0, ValR, ValS, ValT);
			Jacobian[1][0] = x.Differentiate(1, ValR, ValS, ValT);
			Jacobian[2][0] = x.Differentiate(2, ValR, ValS, ValT);
			
			Jacobian[0][1] = y.Differentiate(0, ValR, ValS, ValT);
			Jacobian[1][1] = y.Differentiate(1, ValR, ValS, ValT);
			Jacobian[2][1] = y.Differentiate(2, ValR, ValS, ValT);
			
			Jacobian[0][2] = z.Differentiate(0, ValR, ValS, ValT);
			Jacobian[1][2] = z.Differentiate(1, ValR, ValS, ValT);
			Jacobian[2][2] = z.Differentiate(2, ValR, ValS, ValT);
			
			T LocalDet = Determinant3(Jacobian);
			Determinant[iD + counter * ElementCount] = LocalDet;
			Inverse3(Jacobian); // now invert the jacobian
			counter++;
		}
	}
}

}
