using System;

namespace FEA
{
	public class FiniteVolume
	{
		int[] ElementLocations; // starting point indices of a new element in the array
		int[] NodeIDS; // global ids of all the nodes in each element
		int[] NodeLocations; // starting point indices of a new node in the array
		int[] ElementIDS; // global ids of all the elements that are connected to each node
		public FiniteVolume(int[,] DelaunayTriangulation, int NodeCount) {
			FindElementlocations(DelaunayTriangulation, NodeCount);
			FiniteVolumeBase (NodeCount);
		}
		private void FiniteVolumeBase(int NodeCount)
		{
//			NodeLocations = new int[NodeCount + 1];
//			for (int i = 0; i < NodeIDS.Length; i++) { // loop through each node contained in each element
//				int NodeNumber = NodeIDS[i] + 1;
//				NodeLocations [NodeNumber]++; // count how elements reference that particular node
//			}
//			for (int i = 1; i < NodeLocations.Length; i++) {
//				NodeLocations [i] = NodeLocations [i - 1] + NodeLocations[i]; // sum up each 
//			}
//			var ConnectionCounter = new int[NodeCount];
//			int[] ElementIDS = new int[NodeLocations [NodeLocations.Length - 1]];
//			for (int iElement = 0; iElement < ElementLocations.Length; iElement++) {	
//				int StartPt = ElementLocations [iElement];
//				int EndPt = ElementLocations [iElement + 1];
//				for (int iNode = StartPt; iNode < EndPt; iNode++) {
//					int CurrentNode = NodeIDS [iNode];
//					int StartAddress = NodeLocations [CurrentNode];
//					int offset = StartAddress + ConnectionCounter [CurrentNode];
//					ElementIDS[offset] = iElement;
//					ConnectionCounter[CurrentNode]++;
//				}
//			}
		}
//		public Sparse CreateSparse(int[] ConnectionCount) { // use this function to pre-allocate space for a CSR Compatible Sparse Array
//			
//		}
		private void FindElementlocations(int[,] DelaunayTriangulation, int NodeCount) {
			NodeIDS = new int[DelaunayTriangulation.Length];
			ElementLocations = new int[DelaunayTriangulation.GetLength(0) + 1];
			for (int i = 1; i < ElementLocations.Length; i++) {
				ElementLocations[i] = ElementLocations[i - 1] + DelaunayTriangulation.GetLength(1);
			}
			Buffer.BlockCopy(DelaunayTriangulation, 0, NodeIDS, 0, DelaunayTriangulation.Length * sizeof(int));

		}

	}
}

