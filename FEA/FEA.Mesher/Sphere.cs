using System;
using ManagedCuda.VectorTypes;

namespace FEA.Mesher
{
    public struct Sphere // note that in plane the sphere turns into a circle
    {
        public Sphere(double3 O, double R)
        {
            Radius = R;
            Center = O;
        }
        double Radius;
        double3 Center;
    }
}

