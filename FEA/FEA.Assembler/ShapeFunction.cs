using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FEA
{
    enum ElementType { 
    Tetrahedron,
    Hexahedron
    };
    enum CoordinateSystem {
    Cartesian,
    Natural
    };
    public class ShapeFunction
    {
        public static PolyMatrix Generate(Point LowerBound, Point Upperbound, int Order) {
            double dX = Upperbound.x - LowerBound.x;
            double dY = Upperbound.y - LowerBound.y;
            double dZ = Upperbound.z - LowerBound.z;
            int NodeCount = (int)Math.Pow(Order + 1, 3);
            double[] XValues = new double[Order + 1];
            double[] YValues = new double[Order + 1];
            double[] ZValues = new double[Order + 1];

            for (int idx = 0; idx <= Order; idx++)
            {
                double Fraction = (((double)idx) / (double)Order);
                XValues[idx] = LowerBound.x + dX * Fraction;
                YValues[idx] = LowerBound.y + dY * Fraction;
                ZValues[idx] = LowerBound.z + dZ * Fraction;
            }
            var XPolys = Polynomial.LagrangeInterpolation(XValues);
            var YPolys = Polynomial.LagrangeInterpolation(YValues);
            var ZPolys = Polynomial.LagrangeInterpolation(ZValues);
            var ShapeFunction = new PolyMatrix(1, NodeCount);
            int id = 0;
            for (int i = 0; i <= Order; i++)
            {
                for (int j = 0; j <= Order; j++)
                {
                    for (int k = 0; k <= Order; k++)
                    {
                        ShapeFunction.Data[0, id] = XPolys[i].Convert_3D(0) * YPolys[j].Convert_3D(1) * ZPolys[k].Convert_3D(2);
                        id++;
                    }
                }
            }
                
            return ShapeFunction;
        }
    };
}
