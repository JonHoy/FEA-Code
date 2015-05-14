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
    Cartesian, // xyz space
    Natural // rst space 
    };
    public class ShapeFunction
    {
		#region Quadrilateral Family
		public static PolyMatrix Generate (Point Bound, int Order) {
			return Generate (Bound, Order, Order, Order);
		}
		public static PolyMatrix Generate(Point Bound, int xOrder, int yOrder, int zOrder) {           
			double[] XValues = new double[xOrder + 1];
            double[] YValues = new double[yOrder + 1];
            double[] ZValues = new double[zOrder + 1];
			for (int i = 0; i <= xOrder; i++) {
				XValues[i] = Bound.x * ((double)i) / ((double)xOrder);
			}
			for (int i = 0; i <= yOrder; i++) {
				YValues[i] = Bound.y * ((double)i) / ((double)yOrder);
			}
			for (int i = 0; i <= zOrder; i++) {
				ZValues[i] = Bound.z * ((double)i) / ((double)zOrder);
			}
            var XPolys = Polynomial.LagrangeInterpolation(XValues);
            var YPolys = Polynomial.LagrangeInterpolation(YValues);
            var ZPolys = Polynomial.LagrangeInterpolation(ZValues);
            var ShapeFunction = new PolyMatrix(1, XValues.Length * YValues.Length * ZValues.Length);
            int id = 0;
            for (int i = 0; i <= xOrder; i++)
            {
                for (int j = 0; j <= yOrder; j++)
                {
                    for (int k = 0; k <= zOrder; k++)
                    {
						ShapeFunction.Data[0, id] = XPolys[i].Convert_ND(0) * YPolys[j].Convert_ND(1) * ZPolys[k].Convert_ND(2);
                        id++;
                    }
                }
            }
                
            return ShapeFunction;
        }
		#endregion
    };
}
