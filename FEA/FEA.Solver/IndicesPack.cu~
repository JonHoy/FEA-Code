
// This Kernel takes global ids of the nodes in each element in the array 
// It then uses them to compute the sparse matrix indices of the resulting matrix.
// this step is essential for GPU assembly of the stiffness matrix    

// Keep in mind that this kernel has limits of (2^32 - 1) ~ 4.3 billion node ids

// Call order should be IndicesPack -> Thrust Sort -> Thrust Unique -> IndicesUnpack

extern "C" __global__ void IndicesPack(unsigned int NodesPerElement, unsigned int NumElements, unsigned int* MeshNodes, unsigned long* MatrixEntries) {
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int elementId = x + y + blockDim.x * gridDim.x; // which element we currently are on
	if (elementId < NumElements) { 	// do bounds checking
		unsigned int Offset = elementId * NodesPerElement * NodesPerElement; // where is this start location in Matrix Entries	
		unsigned int id = 0; // where we start before the 
		for (unsigned int i = 0; i < NodesPerElement; i++) {
			for (unsigned int j = 0; j < NodesPerElement; j++) {
				unsigned long x_id = (unsigned long) MeshNodes[elementId * NodesPerElement + i];
				unsigned long y_id = (unsigned long) MeshNodes[elementId * NodesPerElement + j];
				x_id = x_id << 32; // shift x_id to the left by 32 bits
				MatrixEntries[Offset + id] = x_id | y_id; // now combine x_id and y_id side by side to create a packed index number
				id++;
			}	
		}
	}
} 

extern "C" __global__ void IndicesUnpack(unsigned long* MatrixEntries, unsigned long MatrixLength, unsigned int* Rows, unsigned int* Cols) {
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int offset = x + y + blockDim.x * gridDim.x;  // which index we currently are on
	if (offset < MatrixLength) { // range check
		unsigned long RowValue = MatrixEntries[offset];
		RowValue = RowValue >> 32; // shift to the right by 32 bits then cast
		Rows[offset] = (unsigned int) RowValue;
		unsigned long ColValue = MatrixEntries[offset];
		ColValue = (ColValue << 32) >> 32; // shift to the left to knock out left values and then back to the right
		Cols[offset] = ColValue;
	}		

} 
