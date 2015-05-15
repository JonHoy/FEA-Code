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
			// this is for simplex type elements only
			int NodesPerElement = NDelaunay.GetLength (1);
			var ReferenceCount = new int[NodeCount + 1]; // count how many elements reference each node
			// ReferenceCount[i] is how many elements are using global node i + 1 
			for (int i = 0; i < NDelaunay.GetLength(0); i++) {
				for (int j = 0; j < NDelaunay.GetLength(1); j++) {
					ReferenceCount [NDelaunay [i, j] + 1]++; 
				}
			}
			var ConnectivityMatrix = new int[NDelaunay.Length]; // C (not C#) jagged array which stores the ids connected elements to each node 
			for (int i = 1; i < NodeCount + 1; i++) {
				ReferenceCount [i] = ReferenceCount [i] + ReferenceCount [i - 1]; // calculate the starting adress
			}
			var OffsetCounter = new int[NodeCount];
			for (int i = 0; i < NDelaunay.GetLength(0); i++) {
				for (int j = 0; j < NDelaunay.GetLength(1); j++) {
					int Node = NDelaunay [i, j];
					int Element = i;
					OffsetCounter[Node]++;
					int idx = ReferenceCount[Node + 1] - OffsetCounter [Node];
					ConnectivityMatrix [idx] = Element;
				}
			}
			int ElementCount;
			RowsCompressed = new int[NodeCount + 1];
			for (int iNode = 1; iNode < NodeCount + 1; iNode++) {
				ElementCount = ReferenceCount [iNode] - ReferenceCount [iNode - 1];
				int[] NodeGroup = new int[NDelaunay.GetLength(1) *  ElementCount]; 
				int idx = 0;
				for (int iElement = 0; iElement < ElementCount; iElement++) {
					for (int j = 0; j < NDelaunay.GetLength(1); j++) {
						NodeGroup [idx] = NDelaunay[ConnectivityMatrix[ReferenceCount[iNode] + iElement], j];
						idx++;
					}
				}
				RowsCompressed [iNode + 1] = NodeGroup.Distinct().Count() + RowsCompressed[iNode];
			}
			Cols = new int[RowsCompressed [RowsCompressed.Length - 1]];
			for (int i = 0; i < NodeCount; i++) {
				// now add in values for the sparse columns
				ElementCount = RowsCompressed[i + 1] - RowsCompressed[i];
				int[] NodeGroup = new int[NDelaunay.GetLength(1) *  ElementCount]; 
				int idx = 0;
				for (int iElement = 0; iElement < ElementCount; iElement++) {
					for (int j = 0; j < NDelaunay.GetLength(1); j++) {
						NodeGroup [idx] = NDelaunay[ConnectivityMatrix[ReferenceCount[i] + iElement], j];
						idx++;
					}
				}
				NodeGroup = NodeGroup.Distinct().ToArray();
				Buffer.BlockCopy (NodeGroup, 0, Cols, RowsCompressed [i], NodeGroup.Length * sizeof(int));
			}
		}
		public Sparse (int[] Rows, int[] Cols, int[] RowsCompressed) {
			this.Rows = Rows;
			this.Cols = Cols;
			this.RowsCompressed = RowsCompressed;
		}
		public Sparse (Element[] Elements, Point[] Points)
		{
			Initializer (false, Elements, Points);
			Initializer (true, Elements, Points);
		}
		private void Initializer(bool IsAllocated, Element[] Elements, Point[] Points) {
			if (IsAllocated == false)
				RowsCompressed = new int[Points.Length + 1];
			int MaxRowWidth;
			for (int i = 0; i < Elements.Length; i++) {
				MaxRowWidth = Elements[i].Nodes.Length;
				for (int j = 0; j < Elements[i].ConnectedElements.Length; j++) {
					int ConnectionId = Elements [i].ConnectedElements [j];
					MaxRowWidth = MaxRowWidth + Elements [ConnectionId].Nodes.Length;
				}
				var Connections = new List<int>(MaxRowWidth);
				for (int j = 0; j < Elements[i].ConnectedElements.Length; j++) {
					int ConnectionId = Elements [i].ConnectedElements [j];
					Connections.AddRange (Elements [ConnectionId].Nodes.ToList());
				}
				Connections.Distinct();
				if (IsAllocated == false) {
					RowsCompressed [i + 1] = RowsCompressed [i] + Connections.Count;
				} 
				else {
					var Block = Connections.ToArray ();
					Buffer.BlockCopy (Block, 0, Cols, RowsCompressed[i], sizeof(int) * Block.Length);
					for (int j = 0; j < Block.Length; j++) {
						Rows [RowsCompressed [i] + j] = j;
					}
				}

			}
			if (IsAllocated == false) {
				Rows = new int[RowsCompressed [RowsCompressed.Length - 1]];
				Cols = new int[Rows.Length]; 
			}
		}
		public int getIdx(int i, int j) { // return 1-D index number for the sparse 2d array 
			int Start = RowsCompressed [i];
			int End = RowsCompressed [i + 1];
			return Array.BinarySearch (Cols, Start, End - Start, j);
		}  
	}
}

