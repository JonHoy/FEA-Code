using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FEA.Assembler
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
            double[] XValues = new double[NodeCount];
            double[] YValues = new double[NodeCount];
            double[] ZValues = new double[NodeCount];
            int idx = 0;
            for (int i = 0; i <= Order; i++)
            {
                for (int j = 0; j <= Order; j++)
                {
                    for (int k = 0; k <= Order; k++)
                    {
                        XValues[idx] = LowerBound.x + dX * i / Order;
                        YValues[idx] = LowerBound.y + dY * j / Order;
                        ZValues[idx] = LowerBound.z + dZ * k / Order;
                        idx++;
                    }
                }
            }
            var XPolys = Polynomial.LagrangeInterpolation(XValues);
            var YPolys = Polynomial.LagrangeInterpolation(YValues);
            var ZPolys = Polynomial.LagrangeInterpolation(ZValues);
            var ShapeFunction = new PolyMatrix(1, NodeCount);
            for (int j = 0; j < NodeCount; j++)
            {
                ShapeFunction.Data[0, j] = XPolys[j].Convert_3D(0) * YPolys[j].Convert_3D(1) * ZPolys[j].Convert_3D(2);
            }
            return ShapeFunction;
        }
    };
}
