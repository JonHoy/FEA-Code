using System;
using ManagedCuda.VectorTypes;

namespace FEA.Mesher
{
    public struct Line
    {
        public Line(double3 P1, double3 P2)
        {
            var D = P1 - P2;
            if (D.x > 0)
            {
                A = P1;
                B = P2;
            }
            else if (D.x < 0)
            {
                B = P1;
                A = P2;
            }
            else {
                if (D.y > 0)
                {
                    A = P1;
                    B = P2;
                }
                else if (D.y < 0)
                {
                    B = P1;
                    A = P2;
                }
                else {
                    if (D.z > 0)
                    {
                        A = P1;
                        B = P2;
                    }
                    else if (D.z < 0)
                    {
                        B = P1;
                        A = P2;
                    }
                    else {
                        throw new Exception("P1 and P2 must be different real values");
                    }
                }
            }
        }

        public double3 A;
        public double3 B;
    }
}

