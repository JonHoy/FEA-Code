using System;
using System.Linq;
using System.Collections.Generic;

namespace FEA
{
	public class NaturalCoordinate
	{
		int Rank;
		double[,] Map;
		Point[] Points;
		public NaturalCoordinate(double[,] Locations) { 
			Rank = Locations.GetLength(0);
			Map = new double[Locations.GetLength(0), Locations.GetLength(0)];
			for (int j = 0; j < Map.GetLength(1); j++) {
				Map [0, j] = 1;
			}
			for (int i = 1; i < Map.GetLength(0); i++) {
				for (int j = 0; j < Map.GetLength(1); j++) {
					Map [i, j] = Locations [j, i - 1];
				}
			}
			if (Rank == 4) {
				Points = new Point[4];
				for (int i = 0; i < Points.Length; i++) {
					Points[i].x = Locations [i, 0];
					Points[i].y = Locations [i, 1];
					Points[i].z = Locations [i, 2];
				}
			}
		}
		public static double Determinant(double [,] A) {
			double ans;
			int Rows = A.GetLength (0);
			if (Rows == 2)
				ans = A [0, 0] * A [1, 1] - A [0, 1] * A [1, 0];
			else {
				var SubMats = new double[Rows][,];
				for (int k = 0; k < Rows; k++) {
					SubMats[k] = new double[Rows - 1, Rows - 1];
					var IterationRange = new int[Rows - 1];
					int i = 0;
					for (int iPt = 0; iPt < IterationRange.Length; iPt++) {
						if (iPt == k)
							i++;
						IterationRange[iPt] = i;
						i++;
					}
					for (int idx = 0; idx < Rows - 1; idx++) {
						for (int idy = 0; idy < Rows - 1; idy++) {
							SubMats [k] [idx, idy] = A [idx + 1, IterationRange [idx]];
						}
					}
				}
				ans = 0;
				for (int i = 0; i < Rows; i++) {
					double localDeterminant = Determinant(SubMats[i]);
					if (i % 2 == 0)
						ans = ans + localDeterminant;
					else {
						ans = ans - localDeterminant;
					}
				}

			}
			return ans;
		}
		public double IntegrateND(int[] Exponents) {
			double Region = Determinant(Transpose(Map));
			int num = 1;
			for (int i = 0; i < Exponents.Length; i++) {
				num = num * Factorial (Exponents [i]);
			}
			int den = Rank;
			for (int i = 0; i < Exponents.Length; i++) {
				den = den + Exponents [i];
			}
			den = Factorial (den);
			var Multiplier = ((double)num) / ((double)den);
			return Region * Multiplier;
		}
		public double IntegrateVolume(int alpha, int beta, int gamma, int delta) {
			int num = Factorial(alpha)*Factorial(beta)*Factorial(gamma)*Factorial(delta);
			int den = Factorial (alpha + beta + gamma + delta + 3);
			double ans = ((double)((double)num / (double)den)) * Determinant (Transpose (Map));
			return ans;
		}
		public double IntegrateArea(int alpha, int beta, int gamma, int FaceId) { //if faceId = 1 it uses points (2,3,4) 
			int num = Factorial(alpha) * Factorial(beta) * Factorial(gamma);
			int den = Factorial (alpha + beta + gamma + 2);
			double ans = ((double)(num / den)) * CalculateFaceArea (FaceId);
			return ans;
		}
		public double IntegrateLength(int alpha, int beta, int Point1_id, int Point2_id) {
			Point Point1 = Points[Point1_id];
			Point Point2 = Points[Point2_id];
			double Length = CalculateLength(Point1, Point2);
			int num = Factorial (alpha) * Factorial (beta);
			int den = Factorial (alpha + beta + 1);
			double ans = ((double)num / den) * Length;
			return ans;
		}
		public double CalculateFaceArea(int FaceId) {
			// use herons formula
			int id = 0;
			var FacePoints = new Point[3];
			for (int i = 0; i < Points.Length; i++) {
				if (FaceId == i)
					continue;
				FacePoints [id] = Points [i];
				id++;
			}
			double L1 = CalculateLength (FacePoints [0], FacePoints [1]);
			double L2 = CalculateLength (FacePoints [1], FacePoints [2]);
			double L3 = CalculateLength (FacePoints [2], FacePoints [0]);
			double halfP = (L1 + L2 + L3) / 2;
			double ans = Math.Sqrt (halfP * (halfP - L1) * (halfP - L2) * (halfP - L3));
			return ans;
		}
		public Point CalculateFaceUnitNormal(int FaceId) {
			var Edges = new Point[3];
			int id = 0;
			for (int i = 0; i < Points.Length; i++) {
				if (FaceId == i)
					continue;
				Edges [id] = Points [i] - Points [FaceId];
				id++;
			}
			var A = Edges [0] - Edges [2];
			var B = Edges [1] - Edges [2];
			var C = Point.Cross (A, B);
			var dotP = Point.Dot (A, B);
			if (dotP < 0) {
				C = C * -1;
			}
			return C.Normalize ();
		}
		private static int Factorial(int Val) {
			int Ans = 1;
			for (int i = 1; i <= Val; i++) {
				Ans = Ans * i;
			}
			return Ans;
		}
		private static double [,] Transpose(double[,] A) {
			var ans = new double[A.GetLength(1), A.GetLength(0)];
			for (int i = 0; i < A.GetLength(1); i++) {
				for (int j = 0; j < A.GetLength(0); j++) {
					ans [i, j] = A [j, i];
				}
			}
			return ans;
		}
		private double CalculateLength(Point Point1, Point Point2) {
			double dx = Point1.x - Point2.x;
			double dy = Point1.y - Point2.y;
			double dz = Point1.z - Point2.z;
			double Length = Math.Sqrt (dx * dx + dy * dy + dz * dz);
			return Length;
		}

		public static PolyMatrix ConstructInterpolationFunction(int Rank, int InterpolationOrder) {
			// For Line Elements -> Rank = 1, Triangular Elements -> Rank = 2, Tetrahedral Elements -> Rank = 3
			// For linear approximation InterpolationOrder = 2, Quadratic = 3, Cubic = 4, Quintic = 5, etc
			// For reference refer to pp. 282 of Computational Fluid Dynamics 1st Edition by T.J. Chung 
			// step 1 calculate the number of nodes in the element ->  
			int[] Order = new int[Rank + 1];
			for (int i = 0; i < Order.Length; i++) {
				Order [i] = InterpolationOrder + 1;
			}
			var Id = new Index (Order);
			int IterationLength = (int)Math.Pow (InterpolationOrder + 1, Order.Length); // define the maximum number of permutations
			int NodeCount = 0;

			for (int i = 0; i < IterationLength; i++) {
				var Sub = Id.Ind2Sub (i);
				int Tally = Sub.Sum (); // add up each subscript (if they add up to the Interpolation Order + 1 then record them)
				if (Tally == InterpolationOrder) {
					NodeCount++;
				}
			}
			
			double[,] CoordinateVals = new double[NodeCount, Rank + 1];


			int idx = 0;
			for (int i = 0; i < IterationLength; i++) {
				var Sub = Id.Ind2Sub (i);
				int Tally = Sub.Sum (); // add up each subscript (if they add up to the Interpolation Order + 1 then record them)
				if (Tally == InterpolationOrder) {
					for (int j = 0; j < CoordinateVals.GetLength(1); j++) {
						CoordinateVals [idx, j] = ((double)Sub [j]) / ((double)Tally);
					}
					idx++;
				}
			}//
			var B = new PolyMatrix (NodeCount, 1);
			var BOrder = new int[CoordinateVals.GetLength(1)];
			for (int i = 0; i < BOrder.Length; i++) {
				BOrder [i] = 1;
			}
			for (int i = 0; i < NodeCount; i++) {
				B.Data [i,0] = new PolynomialND (BOrder, new double[] { 1 });
				var LocalPolys = new Polynomial[CoordinateVals.GetLength (1)];
				for (int j = 0; j < CoordinateVals.GetLength(1); j++) {
					if (CoordinateVals[i,j] == 0) {
						LocalPolys [j] = new Polynomial (new double[] { 1 });
						continue;
					}
					LocalPolys [j] = new Polynomial (new double[] { 1 });
					int d = (int) (((double)InterpolationOrder) * CoordinateVals[i,j]);
					for (int s = 1; s <= d; s++) {
						double a = ((double)InterpolationOrder) / ((double)s);// ax + b -> {b, a}
						double b = (double)(1 - s) / (((double)s));
						LocalPolys [j] = LocalPolys [j] * new Polynomial (new double[] { b, a });
					}
					B.Data [i, 0] = B.Data [i, 0] * LocalPolys [j].Convert_ND (j, LocalPolys.Length); // multiply the lagrange polynomials
				}

			}
			return B;
		}

	}
}

