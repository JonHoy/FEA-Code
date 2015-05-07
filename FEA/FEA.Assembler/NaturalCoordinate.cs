using System;

namespace FEA
{
	public class NaturalCoordinate
	{
		double[,] Map;
		Point[] Points;
		double Volume;
		public NaturalCoordinate (Point[] Pts)
		{
			this.Points = Pts;
			if (Pts.Length < 2 || Pts.Length > 4)
				throw new Exception ("Number of Points must be between 2 and 4!");
			Map [0, 0] = 1; Map [0, 1] = 1; Map [0, 2] = 1; Map [0, 3] = 1;
			for (int i = 0; i < Pts.Length; i++) {
				Map [1, i] = Pts [i].x;
				Map [2, i] = Pts [i].y;
				Map [3, i] = Pts [i].z;
			}
		}

		private static int Factorial(int Val) {
			int Ans = 1;
			for (int i = 1; i <= Val; i++) {
				Ans = Ans * i;
			}
			return Ans;
		}

	}
}

