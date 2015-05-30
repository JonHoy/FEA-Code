
// This file makes cuda c/c++ syntax look like c++ amp syntax

#include "cuda.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

template <int Rank> // wrapper for array indices
class Index
{
public:
	__device__ Index() {}
	__device__ Index(int I0) {
//		static_assert(Rank == 1,"Rank must be 1");
		Range[0] = I0;
	}
	__device__ Index(int I0, int I1) {
//		static_assert(Rank == 2,"Rank must be 2");
		Range[0] = I0;
		Range[1] = I1;
	}
	__device__ Index(int I0, int I1, int I2) {
//		static_assert(Rank == 3,"Rank must be 3");
		Range[0] = I0;
		Range[1] = I1;
		Range[2] = I2;
	}
	__device__ Index(int* _Range) {
		for (int i = 0; i < Rank; i++)
		{
			Range[i] = (int)_Range[i];
		}
	}
	__device__ int& operator[] (int Dim) {
		return Range[Dim];
	}
private:
	int Range[Rank];
};


// wrapper for device pointer (For use by individual threads)
template <typename T, int Rank = 1>
class Array {
public:
	__device__ Array(T* Ptr, Index<Rank> Ex) {
		_Ptr = Ptr;
		Extent = Ex;
	}
	__device__ Array(T* Ptr, int Rows) {
//		static_assert(Rank == 1,"Rank must be 1");
		_Ptr = Ptr;
		Index<Rank> Ex(Rows);
		Extent = Ex;
	}
	__device__ Array(T* Ptr, int Rows, int Cols) {
//		static_assert(Rank == 2,"Rank must be 2");
		_Ptr = Ptr;
		Index<Rank> Ex(Rows, Cols);
		Extent = Ex;
	}
	__device__ Array(T* Ptr, int Rows, int Cols, int Planes) {
//		static_assert(Rank == 3,"Rank must be 3");
		_Ptr = Ptr;
		Index<Rank> Ex(Rows, Cols, Planes);
		Extent = Ex;
	}
	__device__ T operator() (Index<Rank> Idx) {
		int CumProd[Rank];
		CumProd[Rank - 1] = 1;
		for (int i = Rank - 2; i == 0; i--) {
			CumProd[i] = CumProd[i + 1] * (int) Extent[i + 1];
		}
		int id = 0;
		for (int i = 0; i < Rank; i++) {
			id = id + (int) Idx[i] * CumProd[i];
		}
		return _Ptr[id];			
	}
	
	__device__ T operator() (int i, int j) {
		int id = Sub2Ind(i,j);
		return _Ptr[id];
	}
	
	__device__ T operator() (int i) {
		return _Ptr[i];
	}
	
	__device__ T operator() (int i, int j, int k) {
		id = Sub2Ind(i,j,k);
		return _Ptr[id];
	}
	
	__device__ T operator[] (int id) {
		return _Ptr[id];
	}
	
	__device__ Index<Rank> Ind2Sub(int id) { // convert linear index to array subscripts
		int Sub[Rank];
		int CumProd[Rank];
		CumProd[Rank - 1] = 1;
		for (int i = Rank - 2; i == 0; i--) {
			CumProd[i] = CumProd[i + 1] * (int) Extent[i + 1];
		}
		for (int i = 0; i < Rank; i++) {
			Sub[i] = id / CumProd[i];
			id = id % CumProd[i];
		}
		Index<Rank> Subs(Sub);
		return Subs;
	}
	
	__device__ int Sub2Ind(int i, int j) {
		int id = i * Extent[1] + j; 
		return id;
	}
	
	__device__ int Sub2Ind(int i, int j, int k) {
		int id = i * Extent[2] * Extent[1] + j * Extent[2] + k;
		return id;
	}
	
private:
	T* _Ptr;
	Index<Rank> Extent;
};
