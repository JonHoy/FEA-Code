#pragma once

#include "Vector.cuh"
#include "Triangle.cuh"

template <typename T>
struct Plane {

    __device__ Plane() {}
    __device__ Plane(Triangle<T> Tri) { // gets the
    	Tri.NormalVector();
    	N = Tri.N;
    	// now solve for d using pt A
    	d = -1.0 * Tri.A.Dot(N);
    }
    // http://www.cs.princeton.edu/courses/archive/fall2000/cs426/lectures/raycast/sld017.htm
    __device__ T Intersection(Vector<T> O, Vector<T> D) {
    	T tval = (-1.0 * O.Dot(N) + d) / D.Dot(N);
    	return tval;
    }
    
    Vector<T> N; //(ax + by + cy = d) 
	T d; // Where a = N.x, b = N.y, c = N.z
	
	// 
	__device__ T AboveOrBelow(Vector<T> Pt) {
        T ans = N.Dot(Pt) - d;
        return ans;
	}
	
};
