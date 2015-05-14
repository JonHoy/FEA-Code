using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FEA;
using System.Diagnostics;

namespace Unit_Tests
{
    class Program
    {
        static void Main(string[] args)
		{
			// Indexing
			PrintIndices ();
			TestNaturalCoordinates ();
			// Polynomial Definitions
			#region Polynomial
			// multiplication test
			var p1 = new Polynomial (new double[] { 1, -1 });
			var p2 = p1 * p1;
			var p3 = p2 * p2; 
			Debug.Assert (p3.Evaluate (1) == 0, "A Polynomial must be zero at one of its roots");
			var pminus = p1 - p1;
			Debug.Assert (pminus.Evaluate (1) == 0, "A Polynomial minus itself must be zero");
			var X = new double[] { 1, 2, 3, 4 };
			var LPolys = Polynomial.LagrangeInterpolation (X);
			for (int j = 0; j < LPolys.Length; j++) {
				for (int i = 0; i < LPolys.Length; i++) {
					double Val = LPolys [i].Evaluate (X [j]);
					Val = Math.Round (Val, 4); // Account for Round off error
					if (i == j)
						Debug.Assert (Val == 1, "Local Shape Function Must be equal to 1 at its own Node!");
					if (i != j)
						Debug.Assert (Val == 0, "Local Shape Function Must be equal to 0 at other Nodes!");
				}
			}     
			#endregion
			// Shape Function Definition Test
			#region ShapeFunction 
			var B = new Point ();
			int Order = 2;
			B.x = 1;
			B.y = 1;
			B.z = 1; // unit cube
			var N = ShapeFunction.Generate (B, Order);
			// Definition 1: Ni(xi,yi,zi) = 1
			var NodeCount = Math.Pow ((double)(1 + Order), 3.0);
			var NumPts = (int)NodeCount;
			var XPts = new double[NumPts];
			var YPts = new double[NumPts];
			var ZPts = new double[NumPts];
			int idx = 0;
			for (int i = 0; i <= Order; i++) {
				for (int j = 0; j <= Order; j++) {
					for (int k = 0; k <= Order; k++) {
						XPts [idx] = B.x * ((double)i / (double)Order);
						YPts [idx] = B.y * ((double)j / (double)Order);
						ZPts [idx] = B.z * ((double)k / (double)Order);
						double Value = N.Data [0, idx].Evaluate (new double[] { XPts[idx], YPts[idx], ZPts[idx] });
						Value = Math.Round (Value, 4); // Account for Round off Error
						Debug.Assert (Value == 1, "By Definition this must be one!");
						idx++;
					}
				}
			}          
			// Definition 3: Ni(x,y,z) = 0 if (x,y,z) = Other Node Points 
			#endregion
			#region Natural Coordinate
			var A = new double[4, 4];
			int id = 0;
			for (int i = 0; i < A.GetLength(0); i++) {
				for (int j = 0; j < A.GetLength(1); j++) {
					A [i, j] = id;
					id++;
				}
			}
			var Ans = NaturalCoordinate.Determinant (A);
			Console.WriteLine ("Determininant = {0}", Ans);
			#endregion
		}
		static private void PrintIndices() {
			var X = new Index (new int[] { 5, 10, 15 });
			int Value = 5 * 10 * 15;
			int[] Sub; int idx;
			for (int i = 0; i < Value; i++) {
				Sub = X.Ind2Sub (i);
				idx = X.Sub2Ind (Sub);
				System.Diagnostics.Debug.Assert (i == idx);
			}
		}

		static private void TestNaturalCoordinates() {
			var N = NaturalCoordinate.ConstructInterpolationFunction (1, 1);
			var N2 = NaturalCoordinate.ConstructInterpolationFunction (2, 2);
			var N3 = NaturalCoordinate.ConstructInterpolationFunction (3, 3);
		}
    }
}
