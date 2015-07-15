using System;
using System.Collections.Generic;
using ManagedCuda;
using ManagedCuda.VectorTypes;

namespace FEA.Mesher
{
    public class PointInserter
    {
        public PointInserter(STLReader[] Files, int NumPoints)
        {
            Domains = Files;
            GridCount = Domains.Length;
            int BlockSize = 512;
            TestPointsPerFile = NumPoints;
            if (BlockSize % TestPointsPerFile != 0)
                throw new Exception("NumPoints must be divisible by " + BlockSize.ToString());
            int[] TriangleCounts = new int[GridCount + 1];
            var Maxima = new float3[GridCount];
            var Mimima = new float3[GridCount];
            TriangleCounts[0] = 0;
            for (int i = 0; i < GridCount; i++)
            {
                int LocalCount = TriangleCounts[i] + (int)Files[i].TriangleCount;
                if (LocalCount > BlockSize)
                {
                    throw new Exception("STL File must have no more than " + BlockSize.ToString() + " Triangles");
                }
                TriangleCounts[i + 1] = LocalCount;
                Mimima[i] = STLReader.ToFloat3(Files[i].Extrema.Min);
                Maxima[i] = STLReader.ToFloat3(Files[i].Extrema.Max);
            }
            var Triangles = new TriangleSTL[TriangleCounts[GridCount]];
            int id = 0;
            for (int i = 0; i < GridCount; i++)
            {
                for (int j = 0; j < TriangleCounts[i]; j++) {
                    var LocalTri = Files[i].Triangles[j];
                    Triangles[id] = new TriangleSTL(LocalTri);
                    id++;
                }
            }
            //var DeviceVariables = new Dictionary<string, CudaDeviceVariable<struct>>();
            //DeviceVariables.Add("Triangles",new CudaDeviceVariable<TriangleSTL>(h_Triangles.Length));
            //DeviceVariables.Add("TriangleCounts",new CudaDeviceVariable<int>(GridCount));
            //DeviceVariables.Add("Maxima", new CudaDeviceVariable<float3>(GridCount));
            //DeviceVariables.Add("Minima", new CudaDeviceVariable<float3>(GridCount));

            var d_Triangles = new CudaDeviceVariable<TriangleSTL>(Triangles.Length);
            var d_TriangleCounts = new CudaDeviceVariable<int>(GridCount);
            var d_Minima = new CudaDeviceVariable<float3>(GridCount);
            var d_Maxima = new CudaDeviceVariable<float3>(GridCount);
            var d_Points = new CudaDeviceVariable<float3>(GridCount);

            // copy over to host
            // TODO generate grid on GPU instead of CPU

            var ctx = new CudaContext(CudaContext.GetMaxGflopsDeviceId());
            var PointInPolygonKernel = ctx.LoadKernelPTX("PointInPolygon.ptx", "PointInPolygon");
            var BlockDim = new dim3(BlockSize, 1, 1);
            var GridDim = new dim3(GridCount, 1, 1);

            PointInPolygonKernel.BlockDimensions = BlockDim;
            PointInPolygonKernel.GridDimensions = GridDim;

            PointInPolygonKernel.Run(GridCount,
                NumPoints,
                d_TriangleCounts.DevicePointer,
                d_Maxima.DevicePointer,
                d_Minima.DevicePointer,
                d_Points.DevicePointer);

        }
        int GridCount;
        STLReader[] Domains;
        int TestPointsPerFile;
        /*
        extern "C" __global__ void PointInPolygon(const int Count, // number of files
        const int PointCountPerSTL, // number of test points per stl file 
        int* TriangleCounts, // pointer to jagged array of triangle counts
        Triangle<float>* Triangles, // pointer to jagged array of triangles for each stl 
        Vector<float>* Maxima, // pointer to maximum x-y-z values of each stl
        Vector<float>* Minima, // point to minimum x-y-z values of each stl (Think Bounding Box)
        Vector<float>* Points) // Test Points (values which equal nan are outside polygon)
        */

    }
}

