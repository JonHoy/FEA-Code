
// hard coded versions of a determinant

template <typename T>
__device__ T Determinant4(T A[4][4]);

template <typename T>
__device__ T Determinant3(T A[3][3]);

template <typename T> // A[Rows][Cols]
__device__ T Determinant2(T A[2][2]); 
