#pragma once

#include "Vector.cuh"

template <typename T>
class Plane {

    Vector<T> N; //(ax + by + cy = d) 
	T d; // Where a = N.x, b = N.y, c = N.z
	
	// 
	T AboveOrBelow(Vector<T> Pt) {
        T ans = N.Dot(Pt) - d;
	}
	
}
