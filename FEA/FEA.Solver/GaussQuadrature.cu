
#include "Array.cuh"

template <typename T, int Dims>
__device__ T Integrate(int* Exponents, T* Coefficients, int NumCoeffs) {

// 6 point gaussian quadrature
T Weights[6];
T Locations[6];

Weights[0] = 0.1713244928;
Weights[5] = Weights[0];
Weights[1] = 0.3607615730;
Weights[4] = Weights[1];
Weights[2] = 0.4679139346;
Weights[3] = Weights[2];

Locations[0] = -0.9324695142;
Locations[5] = -Locations[0];
Locations[1] = -0.6612093865;
Locations[4] = -Locations[1];
Locations[2] = -0.2386191860;
Locations[3] = -Locations[2];

// Dims
// int* exponents = [1 1 1	2 4 2	0 0 4]
// A = a0 * x^1 * y^1 * z^1 + a1 * x^2 * y^4 * z^2 + a2 * x^0 * y^0 * z^0 
}
