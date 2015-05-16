using System;
using System.Collections.Generic;
using System.Linq;
namespace FEA
{
	public class Sparse
	{
		// this is a container for the indicies of a sparse matrix not the values
		public int[] Rows;
		public int[] RowsCompressed;
		public int[] Cols;

		public Sparse (int[,] NDelaunay, int NodeCount) {
			HashSet<ulong> mySet = new HashSet<ulong> (); // make a hashset that contains the unique indices of the array. After creation, it can be sorted
			int NRows = NDelaunay.GetLength (0);
			int NCols = NDelaunay.GetLength (1);
			for (int i = 0; i < NRows; i++) {
				for (int j = 0; j < NCols; j++) {
					for (int k = 0; k < NCols; k++) {
						mySet.Add (PackInts (NDelaunay [i, k], NDelaunay [i, j]));
					}
				}
			}

			var myArr = mySet.ToArray<ulong> ();
			Array.Sort (myArr);
			Rows = new int[myArr.Length];
			Cols = new int[myArr.Length];
			int jdx; int idx;
			for (int i = 0; i < Rows.Length; i++) {
				UnpackInts (out idx, out jdx, myArr[i]);
				Rows [i] = idx;
				Cols [i] = jdx;
			}
			// now compress the rows
		}
		public int getIdx(int i, int j) { // return 1-D index number for the sparse 2d array 
			int Start = RowsCompressed [i];
			int End = RowsCompressed [i + 1];
			return Array.BinarySearch (Cols, Start, End - Start, j);
		}

		public static ulong PackInts(int a, int b) {
			var ans = (ulong) a;
			var ansb = (ulong) b;
			ans = ans << 32;
			ans = ans | ansb;
			return ans;
		}
		public static void UnpackInts(out int a, out int b, ulong c) {
			a = (int)(c >> 32);
			ulong bLong = c << 32;
			b = (int)(bLong >> 32);
		}
	}
}

