using System;
using FEA.Assembler;
using System.Collections.Generic;

namespace FEA.Mesher
{
    public class BSpline_Basis_Function
    {
        // using Cox-de Boor formula to calculate basis functions
        // This forms a triangular dependence, see pp 51 An Introduction to NURBS by Rogers
        // This is tricky to implement since adjacent elements of the knot vector can be equal to each other
        public BSpline_Basis_Function(double[] x, int K)
        {
            Polys = new Piecewise[x.Length - 1];
            for (int i = 0; i < Polys.Length; i++)
            {
                Polys[i] = new Piecewise(x[i], x[i + 1], new Polynomial(new double[]{ 1 }));
            }
            int Length = Polys.Length - 1;
            for (int iLevel = 2; iLevel < K + 1; iLevel++)
            {
                for (int i = 0; i < Length; i++)
                {
                    var mpoly1 = new Polynomial(new double[]{ -x[i], 1 });
                    var mult1 = x[i + iLevel - 1] - x[i];
                    if (mult1 != 0)
                        mult1 = 1.0 / mult1;
                    mpoly1 = mpoly1 * mult1;
                    var mpoly2 = new Polynomial(new double[]{ x[i + iLevel], -1 });
                    var mult2 = x[i + iLevel] - x[i + 1];
                    if (mult2 != 0)
                        mult2 = 1.0 / mult2;
                    mpoly2 = mpoly2 * mult2;
                    Polys[i] = Polys[i] * mpoly1 + Polys[i + 1] * mpoly2;
                }
                Length--;
            }
        }
        public Piecewise[] Polys;
    }
}

