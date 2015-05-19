using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FEA.Assembler;

namespace FEA.Physics
{
    public class CFD
    {
        public static double[,] ComputeConvectionMatrix(PolyMatrix N, double u, double v, double w, Point Size) {
            var ConvectionMatrix = N.Transpose() * (u * N.Differentiate(0) + v * N.Differentiate(1) + w * N.Differentiate(2));
            var Zero = new Point(); // point that starts at the origin
            return ConvectionMatrix.Integrate(Zero, Size);
        }

    }
}
