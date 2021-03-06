﻿using System;
using ManagedCuda.VectorTypes;

namespace FEA.Mesher
{
    public class BoundingBox // this represents a cuboid which bounds a 3d surface or collection of surfaces 
    {
        public BoundingBox(double3 _Min, double3 _Max)
        {
            Min = _Min;
            Max = _Max;
        }
        public double3 Min;
        public double3 Max;

//        public TransformationMatrix T { get; private set;} // T represents transformation to the global coordinate system
//        // this tranformation matrix allows for xmin, xmax, ... zmax to represent a different coordinate system
//        // When meshing a volume represented by a collection of surfaces and lines, it is important to know the local extrema of possible nodal values
//        // Additionally, parallel implementations of meshing algorithms will experience better performance if less surfaces are involved
//        // This can only happen if the surface bounding box lays outside of the subdomain bounding box. In other words, only surfaces
//        // inside the region of interest have an impact/influence on the meshing process inside that region.

        public bool Intersects(BoundingBox B) { // If the two boxes have an intersecting region the function returns as true 
            //TODO Implement bounding box intersection
            return true;
        }

        public bool IsInside(double3 A) { // returns true if the point is within the bounding box and not on the edge
            bool Status = false;
            var Aprime = A;
            if ((Aprime.x > Min.x) && (Aprime.y > Min.y) && (Aprime.z > Min.z))
            {
                if ((Aprime.x < Max.x) && (Aprime.y < Max.y) && (Aprime.z < Max.z))
                    Status = true;
            }
            return Status;  
        }

    }
}

