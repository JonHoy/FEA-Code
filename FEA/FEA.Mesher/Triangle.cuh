#pragma once

#include "Vector.cuh"

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
	    // https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
	    // there are three possible outcomes
	    // No intersection (return t = -1)
	    // Point intersection (intersection in the triangle return t > 0)
	    // line intersection (Line is in the plane return t = -2)
	    
	    // construct 3 rays 
	    
	    Vector<T> e1, e2, P, Q, S;
	    T det, inv_det, u, v, t;
	    //Find vectors for two edges sharing V1
	    e1 = B - A;
	    e2 = C - A;
	    //Begin calculating determinant - also used to calculate u parameter
	    P = D.Cross(e2);
	    //if determinant is near zero, ray lies in plane of triangle
	    det = e1.Dot(P);
	    
	    if(det > -EPSILON && det < EPSILON) {
	    	return -1.0;
	    }
	    
	    inv_det = 1.0 / det;
	    //calculate distance from V1 to ray origin
	    S = O - A;
	    //Calculate u parameter and test bound
	    u = S.Dot(P) * inv_det;
	    //The intersection lies outside of the triangle
	    if(u < 0.0 || u > 1.0) {
	    	return -1.0;
	    }
	    //Prepare to test v parameter
	    Q = S.Cross(e1);
	    //Calculate V parameter and test bound
	    v = D.Dot(Q) * inv_det;
	    
	      //The intersection lies outside of the triangle
	    if(v < 0.f || u + v  > 1.f) {
	    	return -1.0;
	    }
	    
	    t = e2.Dot(Q) * inv_det;
	    
	    //ray intersection
	    if (t > EPSILON) {
	    	return t;
	    }
	    // No hit, no win
	    return -1.0;
	    
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


};
