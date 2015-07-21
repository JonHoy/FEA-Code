#pragma once

#include "Vector.cuh"
#include "../FEA.Assembler/Inverse.cuh"

#define EPSILON 0.000001
	
//https://en.wikipedia.org/wiki/STL_(file_format)	
	
template <typename T> 
struct Triangle
{ 
	Vector<T> N; // normal vector
	Vector<T> A; // vertex 1
	Vector<T> B; // vertex 2
	Vector<T> C; // vertex 3

	unsigned char att1; // padding and STL conform for single 
	unsigned char att2; // padding and STL conform for single

    __device__ T Intersection(Vector<T> O, Vector<T> D)
    {
		NormalVector();
		T d = -1.0 * A.Dot(N);
		T t = -1.0 * (O.Dot(N) + d) / D.Dot(N);
		
		Vector<T> Zprime = N;
		Zprime.Normalize();
		Vector<T> Xprime = B - A;
		Xprime.Normalize();
		Yprime = Zprime.Cross(Xprime);
		
		Vector<T> P = O + D*t;
		T px = P.Dot(Xprime);
		T py = P.Dot(Yprime);
		
		Vector<T> CA = C - A;
		Vector<T> BA = B - A;
		
	    return t;
    }
    
    __device__ void NormalVector() { // computes the normal vector in place
        // NOTE for STL Binary, this has already been computed
        Vector<T> AB = A - B;
        Vector<T> BC = B - C;
        N = AB.Cross(BC);
    }
    
    __device__ T Area() {
        Vector<T> BA = B - A;
        Vector<T> CA = C - A;
        
        Vector<T> CrossProd = BA.Cross(CA);
        T Area = CrossProd.Length() / 2.0;
        return Area;
    }
    
    __device__ Vector<T> Centroid() { // This is useful for casting rays
    	Vector<T> Centroid = (A + B + C) / 3.0;
    	return Centroid;
    }


};
