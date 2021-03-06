
// This file tests written cuda code for the Mesher Module

#include "Plane.cuh"
#include "Triangle.cuh"
#include "Vector.cuh"


__device__ int getId() {
	int blockId = blockIdx.z + blockIdx.y * gridDim.z + blockIdx.x * gridDim.y * gridDim.z;
	int threadId = threadIdx.z + threadIdx.y * blockDim.z + threadIdx.x * blockDim.y * blockDim.z;	
	int i = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadId;
	return i;
}

extern "C" __global__ void TestCrossProduct(int Count,
Vector<float>* A,
Vector<float>* B,
Vector<float>* C) {
	int i = getId();
	if (i < Count) {
		C[i] = A[i].Cross(B[i]);
	}
}

extern "C" __global__ void TestDotProduct(int Count,
Vector<float>* A,
Vector<float>* B,
float* C) {
	int i = getId();
	if (i < Count) {
		C[i] = A[i].Dot(B[i]);
	}
}

extern "C" __global__ void TestAdd(int Count,
Vector<float>* A,
Vector<float>* B,
Vector<float>* C) {
	int i = getId();
	if (i < Count) {
		C[i] = A[i] + B[i];
	}
}

extern "C" __global__ void TestSubtract(int Count,
Vector<float>* A,
Vector<float>* B,
Vector<float>* C) {
	int i = getId();
	if (i < Count) {
		C[i] = A[i] - B[i];
	}
}

extern "C" __global__ void TestMultiply(int Count,
Vector<float>* A,
float* B,
Vector<float>* C) {
	int i = getId();
	if (i < Count) {
		C[i] = A[i] * B[i];
	}
}

extern "C" __global__ void TestDivide(int Count,
Vector<float>* A,
float* B,
Vector<float>* C) {
	int i = getId();
	if (i < Count) {
		C[i] = A[i] / B[i];
	}
}


extern "C" __global__ void TestLength(int Count,
Vector<float>* A,
float* B) {
	int i = getId();
	if (i < Count) {
		B[i] = A[i].Length();
	}
}

extern "C" __global__ void TestNormalize(int Count,
Vector<float>* A,
Vector<float>* B) {
	int i = getId();
	if (i < Count) {
		B[i] = A[i];
		B[i].Normalize();
	}
}

extern "C" __global__ void TestTriangleArea(int Count,
Triangle<float>* A,
float* B) {	
	int i = getId();
	if (i < Count) {
		B[i] = A[i].Area();
	}
}

extern "C" __global__ void TestNormalVector(int Count,
Triangle<float>* A) {
	int i = getId();
	if (i < Count) {
		A[i].NormalVector();
	}		
}

extern "C" __global__ void TestPlaneIntersection(int Count,
Plane<float>* A,
Vector<float>* O,
Vector<float>* D,
float* t) {
	int i = getId();
	if (i < Count) {
		t[i] = A[i].Intersection(O[i],D[i]);
	}
} 

extern "C" __global__ void TestTriangleIntersection(int Count,
Triangle<float>* A,
Vector<float>* O,
Vector<float>* D,
float* t) {
	int i = getId();
	if (i < Count) {
		t[i] = A[i].Intersection(O[i],D[i]);
	}
} 
 
extern "C" __global__ void TestCentroidCalculations(int Count,
Triangle<float>* A,
Vector<float>* B)
{
	int i = getId();
	if (i < Count) {
		B[i] = A[i].Centroid();
	}
}

