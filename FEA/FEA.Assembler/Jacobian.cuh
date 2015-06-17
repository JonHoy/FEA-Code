
#include "Polynomial.cuh"

#include "Determinant.cuh"
#include "GaussQuadrature.cuh"

template <typename T, int SizeX, int SizeY = 1, int SizeZ = 1, int IntOrder>
// IntOrder is the number of Gaussian_quadrature coeffients used for integration
class Jacobian{
public:
__device__ Jacobian(Polynomial<T,SizeX,SizeY,SizeZ>& _Poly) { 
// Poly is the 3D polynomial relating (r,s,t) space to (x,y,z) space
// In order for Isoparametric Formulations to exist, this must be used
Poly = _Poly;
}


private:

Quadrature<T, IntOrder> Gauss;
Polynomial<T,SizeX,SizeY,SizeZ> Poly;

T J[3][3];

};