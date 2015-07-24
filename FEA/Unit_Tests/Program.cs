using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using FEA.Assembler;
using FEA.Mesher.IGES;
using FEA.Mesher;
using System.Diagnostics;
using MathNet.Numerics.Data.Matlab;
using MathNet.Numerics.LinearAlgebra;
using ManagedCuda.VectorTypes;

namespace Unit_Tests
{
    class Program
    {
        static void Main(string[] args)
		{
            var Tst = new KernelTests();
            TestMesher ();
			TestAssembler ();
		}

		static private void TestAssembler() {
			TestIndices ();
			TestNaturalCoordinates ();
			TestPackUnpack ();
			TestDeterminant ();
			TestPolynomial ();
            TestPiecewise();
            TestShapeFunction ();
			TestPolyMatrix ();
			TestIsoparametric ();
			TestTrim ();
			TestGeneration ();
			TestSparseGeneration("UniformSpacing.mat");
		}

		static private void TestMesher() {
            TestBasisFunction();
            TestSTLReader();
            Test2dMesher();
		}

        static private void TestPointInPolygon() {
            var STLFile = new STLReader("Cable support hook.stl");
        }

        static private void Test2dMesher() {
            // make a 2 non intersecting, non overlapping boxes put a hole in one of them and make sure the mesher does its job properly.

            var Pts = new List<TriangleNet.Geometry.Vertex>();
            var Lines = new List<TriangleNet.Geometry.IEdge>();
            var Poly = new TriangleNet.Geometry.Polygon();
            // box1
            Poly.Add(new TriangleNet.Geometry.Vertex(0, 0));
            Poly.Add(new TriangleNet.Geometry.Vertex(1, 0));
            Poly.Add(new TriangleNet.Geometry.Edge(0, 1));
            Poly.Add(new TriangleNet.Geometry.Vertex(1, 1));
            Poly.Add(new TriangleNet.Geometry.Edge(1, 2));
            Poly.Add(new TriangleNet.Geometry.Vertex(0, 1));
            Poly.Add(new TriangleNet.Geometry.Edge(2, 3));
            Poly.Add(new TriangleNet.Geometry.Edge(3, 0));
            // box2
            Poly.Add(new TriangleNet.Geometry.Vertex(2, 2));
            Poly.Add(new TriangleNet.Geometry.Vertex(3, 2));
            Poly.Add(new TriangleNet.Geometry.Edge(4, 5));
            Poly.Add(new TriangleNet.Geometry.Vertex(3, 3));
            Poly.Add(new TriangleNet.Geometry.Edge(5, 6));
            Poly.Add(new TriangleNet.Geometry.Vertex(2, 3));
            Poly.Add(new TriangleNet.Geometry.Edge(6, 7));
            Poly.Add(new TriangleNet.Geometry.Edge(7, 4));
            // hole in box1
            Poly.Add(new TriangleNet.Geometry.Vertex(.25, .25));
            Poly.Add(new TriangleNet.Geometry.Vertex(.75, .25));
            Poly.Add(new TriangleNet.Geometry.Edge(8, 9));
            Poly.Add(new TriangleNet.Geometry.Vertex(.75, .75));
            Poly.Add(new TriangleNet.Geometry.Edge(9, 10));
            Poly.Add(new TriangleNet.Geometry.Vertex(.25, .75));
            Poly.Add(new TriangleNet.Geometry.Edge(10, 11));
            Poly.Add(new TriangleNet.Geometry.Edge(11, 8));

            TriangleNet.IO.TriangleWriter.WritePoly(Poly, "MultiplePolyTest.poly");
        }

		static private void TestPhysics() {}

		static private void TestIndices() {
			var X = new Index (new int[] { 5, 10, 15 });
			int Value = 5 * 10 * 15;
			int[] Sub; int idx;
			for (int i = 0; i < Value; i++) {
				Sub = X.Ind2Sub (i);
				idx = X.Sub2Ind (Sub);
				System.Diagnostics.Debug.Assert (i == idx);
			}
		}

		static private void PrintNaturalCoordinates(PolyMatrix N) {
			for (int iNode = 0; iNode < N.Data.Length; iNode++) {
				var Id = new Index (N.Data [iNode,0].Order);
				string ScreenOutput = "N" + iNode.ToString () + " = ";
				for (int jCoeff = 0; jCoeff < N.Data[iNode,0].Coefficients.Length; jCoeff++) {
					double a = N.Data [iNode,0].Coefficients [jCoeff];
					if (a != 0.0) {
						var Sub = Id.Ind2Sub(jCoeff);
						ScreenOutput = ScreenOutput + a.ToString("F2") ;
						for (int iSub = 0; iSub < Sub.Length; iSub++) {
							if (Sub [iSub] > 1) {
								ScreenOutput = ScreenOutput + "L" + iSub.ToString () + "^" + Sub [iSub] + "*";
							}
							else if (Sub[iSub] == 1) {
									ScreenOutput = ScreenOutput + "L" + iSub.ToString () + "*";
							}
						}
						ScreenOutput = ScreenOutput.Remove (ScreenOutput.Length - 1);
						ScreenOutput = ScreenOutput + " + "; 
					}
				}
				Console.WriteLine(ScreenOutput.Remove(ScreenOutput.Length - 3));
			}
		}

		static private void TestNaturalCoordinates() {
			var N = NaturalCoordinate.ConstructInterpolationFunction (1, 1);
			var N2 = NaturalCoordinate.ConstructInterpolationFunction (2, 2);
			var N3 = NaturalCoordinate.ConstructInterpolationFunction (3, 2);
			//Console.WriteLine ("1D Order Linear Interpolation");
			//PrintNaturalCoordinates (N);
			//Console.WriteLine ("2D Order Quadratic Interpolation");
			//PrintNaturalCoordinates (N2);
			//Console.WriteLine ("3D Order Cubic Interpolation");
			//PrintNaturalCoordinates (N3);
		}

		static private void TestSparseGeneration(string Path) {
			var UniformMat = MatlabReader.Read<double> (Path,"Tri");
			var NodeCount = MatlabReader.Read<double> (Path,"NodeCount");
			var Count = NodeCount.ToArray ();
			var Delaunay = new int[UniformMat.RowCount, UniformMat.ColumnCount];
			for (int i = 0; i < Delaunay.GetLength(0); i++) {
				for (int j = 0; j < Delaunay.GetLength(1); j++) {
					Delaunay [i, j] = (int)UniformMat [i, j] - 1;
				}
			}
			var myArray = new Sparse (Delaunay, (int)Count[0,0]);
		}

		static private void TestPackUnpack() {
			int aout; int bout;
			int a = 511;
			int b = 234400;

			ulong c = Sparse.PackInts (a, b);
			Sparse.UnpackInts (out aout, out bout, c);
			Debug.Assert (a == aout && b == bout);
		}

		static private void TestDeterminant() {
			var A = new double[4, 4];
			int id = 0;
			for (int i = 0; i < A.GetLength(0); i++) {
				for (int j = 0; j < A.GetLength(1); j++) {
					A [i, j] = id;
					id++;
				}
			}
			var Ans = NaturalCoordinate.Determinant (A);
			Debug.Assert (Ans == 0);
		}

		static private void TestPolynomial() {
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
		}

        static private void TestPiecewise() {
            var x = new double[]{ -2.0, 0, 2 };
            var p1 = new Polynomial(new double[]{ -4 });
            var p2 = new Polynomial(new double[]{ 4 });
            var myPoly = new Piecewise(x, new Polynomial[]{ p1, p2 });
            Debug.Assert(myPoly.Evaluate(-2) == -4);
            Debug.Assert(myPoly.Evaluate(-1) == -4);
            Debug.Assert(myPoly.Evaluate(0) == -4);
            Debug.Assert(myPoly.Evaluate(1) == 4);
            Debug.Assert(myPoly.Evaluate(2) == 4);
            myPoly = myPoly * 2;
            Debug.Assert(myPoly.Evaluate(-2) == -8);
            Debug.Assert(myPoly.Evaluate(-1) == -8);
            Debug.Assert(myPoly.Evaluate(0) == -8);
            Debug.Assert(myPoly.Evaluate(1) == 8);
            Debug.Assert(myPoly.Evaluate(2) == 8);
            myPoly = myPoly * myPoly;
            Debug.Assert(myPoly.Evaluate(-2) == 64);
            Debug.Assert(myPoly.Evaluate(-1) == 64);
            Debug.Assert(myPoly.Evaluate(0) == 64);
            Debug.Assert(myPoly.Evaluate(1) == 64);
            Debug.Assert(myPoly.Evaluate(2) == 64);

            var myPoly2 = new Piecewise(-4, 2, p1);
            var myPoly3 = new Piecewise(-2, 4, p2);
            var myPoly4 = myPoly2 + myPoly3;
            Debug.Assert(myPoly4.Evaluate(-100) == 0);
            Debug.Assert(myPoly4.Evaluate(100) == 0);
            Debug.Assert(myPoly4.Evaluate(0) == 0);
            Debug.Assert(myPoly4.Evaluate(-3) == -4);
            Debug.Assert(myPoly4.Evaluate(3) == 4);
        }

        static private void TestNURBS(IGSReader File) {
            foreach (var Curve in File.Curves)
            {
                var Vals = Curve.Evaluate(100);
                Debug.Assert(Vals.Length == 100);
            }
            int Nurbid = 0;
            foreach (var Surface in File.Surfaces)
            {
                int id = 0;

                int Rows = 10;
                var Vals = Surface.Evaluate(Rows,Rows);
                var Raw = new double[Rows * Rows * 3];
                for (int i = 0; i < Rows; i++)
                {
                    for (int j = 0; j < Rows; j++) {
                        Raw[3 * id] = Vals[i, j].x;
                        Raw[3 * id + 1] = Vals[i, j].y;
                        Raw[3 * id + 2] = Vals[i, j].z;
                        id++;
                    }
                }
                var RawBytes = new byte[Raw.Length * sizeof(double) / sizeof(byte)];
                Buffer.BlockCopy(Raw, 0, RawBytes, 0, RawBytes.Length);
                using (BinaryWriter writer = new BinaryWriter(System.IO.File.Open("NURB" + Nurbid + ".dat", FileMode.Create)))
                {
                    writer.Write(RawBytes);
                }
                Nurbid++;
            }



        }

        static private void TestSTLReader() {
            var Part = new STLReader("Cable support hook.stl");
            int MaxCount = 512;
            var SubDivisions = Part.RecursiveSplit(MaxCount);
            foreach (var item in SubDivisions)
            {
                if (item.TriangleCount > MaxCount)
                    throw new Exception("This must be less than " + MaxCount.ToString() + " Triangles");
            }
            int NumPoints = 512 * 10000;
            var Mesher = new PointInserter(SubDivisions.ToArray(), NumPoints);
        }

        static private void TestBasisFunction() {
            // Unit Test Case Taken from Example 3.3 pp 58 An Introduction to NURBS by Rogers

            // Calculate the five third order basis functions for the following knot vector
            // [X] = [0 0 0 1 1 3 3 3]
            var KnotVector1 = new double[]{ 0, 0, 0, 1, 1, 3, 3, 3 };
            var N = new BSpline_Basis_Function(KnotVector1, 3);
            var TestVals = new double[]{ 0, 0.5, 1, 2, 2.9 };
            for (int i = 0; i < TestVals.Length; i++)
            {
                if (TestVals[i] >= 0 && TestVals[i] < 1)
                {
                    Approx_Assert(N.Polys[0].Evaluate(TestVals[i]) , Math.Pow(1.0 - TestVals[i], 2));
                    Approx_Assert(N.Polys[1].Evaluate(TestVals[i]) , 2.0*TestVals[i]*(1.0 - TestVals[i]));
                    Approx_Assert(N.Polys[2].Evaluate(TestVals[i]) , TestVals[i] * TestVals[i]);
                    Approx_Assert(N.Polys[3].Evaluate(TestVals[i]) , 0);
                    Approx_Assert(N.Polys[4].Evaluate(TestVals[i]) , 0);

                }
                else if (TestVals[i] >= 1 && TestVals[i] < 3)
                {
                    Approx_Assert(N.Polys[0].Evaluate(TestVals[i]) , 0);
                    Approx_Assert(N.Polys[1].Evaluate(TestVals[i]) , 0);
                    Approx_Assert(N.Polys[2].Evaluate(TestVals[i]) , (Math.Pow(3.0 - TestVals[i],2)/4.0));
                    Approx_Assert(N.Polys[3].Evaluate(TestVals[i]) , ((3 - TestVals[i]) * (TestVals[i] - 1.0) / 2.0));
                    Approx_Assert(N.Polys[4].Evaluate(TestVals[i]) , (Math.Pow(TestVals[i] - 1.0,2)/4.0));
                    }
            }
            // we must also do checksums to see that they add up to 1
            for (int i = 0; i < TestVals.Length; i++)
            {
                double CheckSum = 0;
                for (int j = 0; j < 5; j++) {
                    CheckSum += N.Polys[j].Evaluate(TestVals[i]);
                }
                Approx_Assert(CheckSum , 1.0);
            }
        }

        static private void Approx_Assert(double Cond1, double Cond2, double ErrorTolerance = 1.0e-7) {
            if (Math.Abs( Cond1 - Cond2) > ErrorTolerance)
                throw new Exception("Failed Test");

        }

		static private void TestShapeFunction() {
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
		}

		static private void TestPolyMatrix() {
			var N = NaturalCoordinate.ConstructInterpolationFunction (2, 2);
			var NTT = N.Transpose ().Transpose ();
			var NNew = N.Integrate (0).Differentiate(0);
			for (int i = 0; i < N.Rows; i++) {
				for (int j = 0; j < N.Cols; j++) {
					int Length = N.Data [i, j].Coefficients.Length;
					for (int k = 0; k < Length; k++) {
						if (N.Data [i, j].Coefficients [k] != NTT.Data [i, j].Coefficients [k])
							throw new Exception ("This should be the same!");
						if (N.Data [i, j].Coefficients [k] != NNew.Data [i, j].Coefficients [k])
							throw new Exception ("This should be the same!");
					}
				}
			}
		}

		static private void TestIsoparametric() {
			Isoparametric.WriteToCppFiles ();
		}

		static private void PrintArray(double[,] A) {
			Console.WriteLine ("A = ");
			for (int i = 0; i < A.GetLength(0); i++) {
				string Val = "[ ";
				for (int j = 0; j < A.GetLength(1); j++) {
					string ValString = A [i, j].ToString ();
					int Len = 25;
					if (ValString.Length > Len)
						ValString = ValString.Substring (0, Len);
					if (ValString.Length < Len)
						ValString = ValString.PadRight (Len);
					Val = Val + ValString + " ";
				}
				Val = Val + "]";
				Console.WriteLine (Val);	
			}
		}
		static private void CheckCols(double [,] A) {
			for (int i = 0; i < A.GetLength(0); i++) {
				double sum = 0;
				for (int j = 0; j < A.GetLength(1); j++) {
					sum = sum + A [i, j];
				}
				Debug.Assert (sum == 1);
			}
		}

		static private void TestTrim() {
			var myExp = new SymbolicExpression ("(((ax + b)))");
			var myExp2 = new SymbolicExpression ("(ax + b)");
			Debug.Assert (myExp.Expression == myExp2.Expression);
		}
		static private void TestGeneration() {
			var myMat = new SymbolicMatrix (16, 16);
		}

    }
}
