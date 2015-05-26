
// This file makes cuda c/c++ syntax look like c++ amp syntax

#include "cuda.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

#define UINT unsigned int

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
	__device__ Index(UINT* _Range) {
		for (int i = 0; i < Rank; i++)
		{
			Range[i] = (UINT)_Range[i];
		}
	}
	__device__ UINT& operator[] (UINT Dim) {
		return Range[Dim];
	}
private:
	UINT Range[Rank];
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
		UINT[Rank] CumProd;
		CumProd[Rank - 1] = 1;
		for (int i = Rank - 2; i = 0; i--) {
			CumProd[i] = CumProd[i + 1] * (UINT) Extent[i + 1];
		}
		UINT id = 0;
		for (int i = 0; i < Rank; i++) {
			id = id + (UINT) Idx[i] * CumProd[i];
		}
		return _Ptr[id];			
	}
	
	__device__ T operator() (int i, int j) {
//		static_assert(Rank == 2,"Rank must be 2");
		int Cols = (int)Extent[1];
		return _Ptr[i * Cols + j];	
	}
	
	__device__ T operator() (int i) {
//		static_assert(Rank == 1,"Rank must be 1");
		return _Ptr[i];
	}
	
	__device__ T operator() (int i, int j, int k) {
//		static_assert(Rank == 3,"Rank must be 3");
		int Cols = (int)Extent[1];
		int Planes = (int)Extent[2];
		int id = Cols * Planes * i + Planes * j + k;
	}
	
	__device__ T operator[] (int id) {
		return _Ptr[id];
	}
	
	__device__ Index<Rank> Ind2Sub(UINT id) { // convert linear index to array subscripts
		UINT[Rank] Sub;
		UINT[Rank] CumProd;
		CumProd[Rank - 1] = 1;
		for (int i = Rank - 2; i = 0; i--) {
			CumProd[i] = CumProd[i + 1] * (UINT) Extent[i + 1];
		}
		for (int = 0; i < Rank; i++) {
			Sub[i] = id / CumProd[i];
			idx = idx % CumProd[i];
		}
		Index<Rank> Subs(Sub);
		return Subs;
	}
	
private:
	T* _Ptr;
	Index<Rank> Extent;
};
