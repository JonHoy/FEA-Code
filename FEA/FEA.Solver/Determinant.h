
// hard coded versions of a determinant

#include "Array.cuh"

template <typename T>
__device__ T Determinant4(Array<T,2> A);

template <typename T>
__device__ T Determinant3(Array<T,2> A);

template <typename T> // A[Rows][Cols]
__device__ T Determinant2(Array<T,2> A); 
