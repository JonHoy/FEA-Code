using System;
using System.Collections.Generic;
using System.Linq;
using ManagedCuda;
using ManagedCuda.VectorTypes;

namespace FEA.Mesher
{
    public class PointInserter
    {
        public PointInserter(STLReader[] _Domains)
        {
            Domains = _Domains;
            GridCount = Domains.Length;
           
        }
        private void CheckInput(int NumPoints) {
            
        }

        public float3[] GetPointsGPU(int NumPoints) {
            int BlockSize = 512;
            if (NumPoints % BlockSize != 0)
                throw new Exception("NumPoints must be divisible by " + BlockSize.ToString());
            int[] TriangleCounts = new int[GridCount + 1];
            var Maxima = new float3[GridCount];
            var Minima = new float3[GridCount];
            TriangleCounts[0] = 0;
            for (int i = 0; i < GridCount; i++)
            {
                int LocalCount = TriangleCounts[i] + (int)Domains[i].TriangleCount;
                if (Domains[i].TriangleCount > BlockSize)
                {
                    throw new Exception("STL File must have no more than " + BlockSize.ToString() + " Triangles");
                }
                TriangleCounts[i + 1] = LocalCount;
                Minima[i] = STLReader.ToFloat3(Domains[i].Extrema.Min);
                Maxima[i] = STLReader.ToFloat3(Domains[i].Extrema.Max);
            }
            var Triangles = new TriangleSTL[TriangleCounts[GridCount]];
            int id = 0;
            for (int i = 0; i < GridCount; i++)
            {
                for (int j = 0; j < TriangleCounts[i]; j++) {
                    var LocalTri = Domains[i].Triangles[j];
                    Triangles[id] = new TriangleSTL(LocalTri);
                    id++;
                }
            }

            var ctx = new CudaContext(1);
            var DeviceInfo = ctx.GetDeviceInfo();
            var d_Triangles = new CudaDeviceVariable<TriangleSTL>(Triangles.Length);
            var d_TriangleCounts = new CudaDeviceVariable<int>(GridCount);
            var d_Minima = new CudaDeviceVariable<float3>(GridCount);
            var d_Maxima = new CudaDeviceVariable<float3>(GridCount);
            var d_Points = new CudaDeviceVariable<float3>(GridCount * NumPoints);
            var h_Points = new float3[GridCount * NumPoints];
            var rng = new Random(0); // use a sequence that is repeatable over and over again
            for (int i = 0; i < GridCount * NumPoints; i++)
            {
                h_Points[i].x = (float)rng.NextDouble();
                h_Points[i].y = (float)rng.NextDouble();
                h_Points[i].z = (float)rng.NextDouble();
            }
            int ctr = 0;
            for (int i = 0; i < GridCount; i++)
            {
                for (int j = 0; j < NumPoints; j++) {
                    h_Points[ctr].x = Minima[i].x + h_Points[ctr].x * (Maxima[i].x - Minima[i].x);
                    h_Points[ctr].y = Minima[i].y + h_Points[ctr].y * (Maxima[i].y - Minima[i].y);
                    h_Points[ctr].z = Minima[i].z + h_Points[ctr].z * (Maxima[i].z - Minima[i].z);
                    ctr++;
                }
            }
            d_Points = h_Points;
            d_Triangles = Triangles;
            d_TriangleCounts = TriangleCounts;
            d_Minima = Minima;
            d_Maxima = Maxima;
            // copy over to host
            // TODO generate grid on GPU instead of CPU


            var PointInPolygonKernel = ctx.LoadKernelPTX("PointInPolygon.ptx", "PointInPolygon");
            var BlockDim = new dim3(BlockSize, 1, 1);
            var GridDim = new dim3(GridCount, 1, 1);

            PointInPolygonKernel.BlockDimensions = BlockDim;
            PointInPolygonKernel.GridDimensions = GridDim;

            PointInPolygonKernel.Run(GridCount,
                NumPoints,
                d_TriangleCounts.DevicePointer,
                d_Triangles.DevicePointer,
                d_Maxima.DevicePointer,
                d_Minima.DevicePointer,
                d_Points.DevicePointer);
            h_Points = d_Points;

            return h_Points; // TODO Fix this to remove bad points

        }

        public double3[] GetPointsCPU(int NumPoints) {
            var TestVals = GenerateRandomPoints(NumPoints * GridCount);
            for (int iDomain = 0; iDomain < Domains.Length; iDomain++)
            {
                for (int jPt = 0; jPt < NumPoints; jPt++)
                {
                    int id = iDomain * NumPoints + jPt;
                    TestVals[id] = (Domains[iDomain].Extrema.Max - Domains[iDomain].Extrema.Min) * TestVals[id] + Domains[iDomain].Extrema.Min;
                    bool InsideStatus = Domains[iDomain].InsideOrOutside(TestVals[id]);
                    if (InsideStatus == false)
                    {
                        TestVals[id] = new double3(double.NaN);
                    }
                }
            }
            TestVals = TestVals.Where(val => !double.IsNaN(val.x)).ToArray();
            return TestVals;
        }

        private double3[] GenerateRandomPoints(int NumPoints) {
            var Ans = new double3[NumPoints];
            var Rng = new Random(0); // make the sequence repeatable between identical calls
            for (int i = 0; i < Ans.Length; i++)
            {
                Ans[i] = new double3();
                Ans[i].x = Rng.NextDouble();
                Ans[i].y = Rng.NextDouble();
                Ans[i].z = Rng.NextDouble();
            }
            return Ans;
        }

        int GridCount;
        STLReader[] Domains;
        int TestPointsPerFile;
        /*
        extern "C" __global__ void PointInPolygon(const int Count, // number of Domains
        const int PointCountPerSTL, // number of test points per stl file 
        int* TriangleCounts, // pointer to jagged array of triangle counts
        Triangle<float>* Triangles, // pointer to jagged array of triangles for each stl 
        Vector<float>* Maxima, // pointer to maximum x-y-z values of each stl
        Vector<float>* Minima, // point to minimum x-y-z values of each stl (Think Bounding Box)
        Vector<float>* Points) // Test Points (values which equal nan are outside polygon)
        */

    }
}

