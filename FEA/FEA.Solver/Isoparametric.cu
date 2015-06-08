
#include "Polynomial.cuh"
#include "Inverse.cuh"
#include "Determinant.cuh"
#include "CoefficientMatrix.cuh"

#define Size 8
#define Dims 3

typedef float T;

extern "C" __global__ void JacobianDeterminant(
const int ElementCount, // how many elements are on gpu   
int* NodeIds,
T* X, // x coordinates of each of the nodes
T* Y, // y coordinates of each of the nodes
T* Z, // z coordinates of each of the nodes
T* Determinant) { // determinants of the jacobian at locations specified by gauss quadrature  

// X is a Matrix of (Size x ElementCount

CoefficientMatrix<T,8,3> C();

T xU[Size];
T yU[Size];
T zU[Size];

T xA[Size];
T yA[Size];
T zA[Size];

int iD = blockIdx.x * blockDim.x + threadIdx.x;

for (int i = 0; i < Size; i++) {
	int Location = NodeIds[iD * Size + i];
	xU[i] = X[Location];
	yU[i] = Y[Location];
	zU[i] = Z[Location];
}		

C.GetAlpha( xA, xU);
C.GetAlpha( yA, yU);
C.GetAlpha( zA, zU);

}
