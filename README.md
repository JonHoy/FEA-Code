

// The end goal is to have the process as follows

// Step 1: Load the geometry data object representing a CAD model onto the GPU
	// - STL, IGES, or STEP Files
// Step 2: Create a Base Finite Element Mesh of the Geometry on the GPU
	// - Research unstructured grid generation on the gpu
// Step 3: Pre-Compute the resulting sparse index entiries on the GPU
	// 
// Step 4: Assemble the Stiffness Matrix on the GPU in place on the GPU using Atomics
	//
// Step 5: Solve the resulting Linear System (Use multigrid + domain decomposition)

// Step 6: Refine the mesh where needed, go back to step 3, exit when this process converges

// To deal with memory issues, for each stage of the process, if the available memory on the gpu is not enough, chunk the data into pieces
// Have it so that the copy engine across PCI-Express bus and the kernel compute engine are running conncurrently so that maximum utiization is acheived
// Keep in mind that the following generic structure must be utilized:

// Can run on a cluster via MPI or on a single workstation OpenMPI

// Supports Mutliple GPU-Acceleration per node and future

// Cross operating system compatible (Windows, Linux, Mac)

// Cross hardware compatible CPU, GPU, Coprocessor (Intel, AMD, NVIDIA) -OpenCL -OpenMP
