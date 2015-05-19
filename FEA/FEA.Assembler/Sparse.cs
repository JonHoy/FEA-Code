using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
namespace FEA.Assembler
{
	public unsafe class Sparse
	{
		// this is a container for the indicies of a sparse matrix not the values
		public int[] Rows;
		public int[] RowsCompressed;
		public int[] Cols;
		[DllImport ("libFEA")]
		private static extern int getpid();
		[DllImport ("libFEA")]
		private static extern ulong STL_Unique(ulong* Ptr, ulong length);
		public Sparse (int[,] NDelaunay, int NodeCount) {
			GenerateCSR (NDelaunay, NodeCount);
//			HashSet<ulong> mySet = new HashSet<ulong> (); // make a hashset that contains the unique indices of the array. After creation, it can be sorted
//			int NRows = NDelaunay.GetLength (0);
//			int NCols = NDelaunay.GetLength (1);
//			for (int i = 0; i < NRows; i++) {
//				for (int j = 0; j < NCols; j++) {
//					for (int k = 0; k < NCols; k++) {
//						mySet.Add (PackInts (NDelaunay [i, k], NDelaunay [i, j]));
//					}
//				}
//			}
//
//			var myArr = mySet.ToArray<ulong> ();
//			Array.Sort (myArr);
//			Rows = new int[myArr.Length];
//			Cols = new int[myArr.Length];
//			int jdx; int idx;
//			for (int i = 0; i < Rows.Length; i++) {
//				UnpackInts (out idx, out jdx, myArr[i]);
//				Rows [i] = idx;
//				Cols [i] = jdx;
//			}
//			// now compress the rows
		}

		private static void GenerateCSR(int[,] NDelaunay, int NodeCount) {
			ulong[] IndicesList = new ulong[NDelaunay.LongLength * NDelaunay.GetLongLength (1)];
			int NRows = NDelaunay.GetLength (0);
			int NCols = NDelaunay.GetLength (1);
			long idx = 0;
			for (int i = 0; i < NRows; i++) {
				for (int j = 0; j < NCols; j++) {
					for (int k = 0; k < NCols; k++) {
						IndicesList[idx] = PackInts (NDelaunay [i, k], NDelaunay [i, j]);
						idx++;
					}
				}
			}
			ulong endPt;
			HashSet<ulong> mySet = new HashSet<ulong> (IndicesList);
			IndicesList = mySet.ToArray ();
			int[] Cols = new int[IndicesList.Length];
			int[] Rows = new int[IndicesList.Length];
			int id_i; int id_j;
			for (int i = 0; i < IndicesList.Length; i++) {
				UnpackInts(out id_i, out id_j, IndicesList[i]);
				Rows [i] = id_i;
				Cols [i] = id_j;
			}
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

