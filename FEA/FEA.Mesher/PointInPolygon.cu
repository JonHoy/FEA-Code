// Point in Polygon

// high level cuda routine that implements point in polygon

// Essentially, this is the node insertion stage of meshing,

// By generating a grid of test points we can insert them into the mesh as nodes if they pass the point in polygon test.

// From a high level view (the original STL File/ Files are broken up into subparts until they reach a certain triangle mesh count (enough to fit in gpu cache)

// These STL files are then passed to a single block of threads with a shared memory cache that can fit them

// TODO Study performance tradeoff of processing many small stl files to a few big ones (To break it up, the performance takes a hit initially, however once broken up into small pieces (Triangle count wise) less triangles need to be looped over for point in polygon

#include "Vector.cuh"
#include "Triangle.cuh"

#define BlockSize 512

extern "C" __global__ void PointInPolygon(const int Count, // number of files
const int PointCountPerSTL, // number of test points per stl file 
int* TriangleCounts, // pointer to jagged array of triangle counts
Triangle<float>* Triangles, // pointer to jagged array of triangles for each stl 
Vector<float>* Maxima, // pointer to maximum x-y-z values of each stl
Vector<float>* Minima, // point to minimum x-y-z values of each stl (Think Bounding Box)
Vector<float>* Points) // Test Points (values which equal nan are outside polygon)
{
	__shared__ Triangle<float> SharedTriangles[BlockSize]; // cache for high bandwidth
	int blockId = blockIdx.z + blockIdx.y * gridDim.z + blockIdx.x * gridDim.y * gridDim.z;
	int threadId = threadIdx.z + threadIdx.y * blockDim.z + threadIdx.x * blockDim.y * blockDim.z;
	int PointCountPerThread = PointCountPerSTL / BlockSize;
	int Offset = blockId * PointCountPerSTL + threadId * PointCountPerThread; // Offset Location for the testpoint
	
	// make sure that only blocks with data get processed 
	if (blockId < Count) {

		int TriangleCount = TriangleCounts[blockId + 1] - TriangleCounts[blockId]; // count the number of triangles in the stl file	

		Vector<float> Origin; // we will cast 3 rays (X dir, Y dir, and Z dir) ) which have an origin outside of the polygon 
		Vector<float> Direction; // Direction is the vector difference of the Test Point and the Origin
		// Ray Origin + Direction*t -> t = 1 at the testpoint
		Vector<float> TestPoint;
		
		Vector<float> Max = Maxima[blockId];
		Vector<float> Min = Minima[blockId];
		if (threadId < TriangleCount)
			SharedTriangles[threadId] = Triangles[TriangleCounts[blockId] + threadId]; // save to shared memory 
		
		__syncthreads(); // sync the threads so that the shared memory is all there and no race conditions occur
		
		for (int i = 0; i < PointCountPerThread; i++)
		{
			TestPoint = Points[i + Offset];
			Origin.x = Min.x;
			Origin.y = TestPoint.y;
			Origin.z = TestPoint.z;
			Direction = TestPoint - Origin;
			
			// TODO add in 6 test rays from {-X, +X, -Y, +Y, -Z, +Z} sides
			
			int AboveCount = 0; // number of ray triangle intersections above the test point
			int BelowCount = 0; // number of ray triangle intersections below the test point
			for (int j = 0; j < TriangleCount; j++) {
				Triangle<float> CurrentTriangle = SharedTriangles[j];
				float t = CurrentTriangle.Intersection(Origin, Direction); // find intersection point if it exists
				if (t > 1.0)
					AboveCount++;
				else if (t != -1.0 && t > 0.0)
					BelowCount++;			
			}
			
			if (BelowCount % 2 == 0) {
				TestPoint.x = 0;
				TestPoint.y = 0;
				TestPoint.z = 0;
				Points[i + Offset] = TestPoint; // re-assign the erased point back
				// point is outside the polygon 
			}	
				
				
		}
		 
	}		
}
