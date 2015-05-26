
// cuda version of ND-polynomial

#include "Array.cuh"

template <typename T, int Rank = 1>
class Polynomial : public Array {
public:
	__device__ Polynomial(T* Coeffs, int Order);
	__device__ Integrate(int Dim = 0);
	__device__ Differentiate(int Dim = 0);
	__device__ T Evaulate(T Value, int Dim = 0);
private:
	
};

