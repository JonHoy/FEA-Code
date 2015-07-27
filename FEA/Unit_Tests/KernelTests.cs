using System;
using ManagedCuda;
using ManagedCuda.VectorTypes;
using ManagedCuda.BasicTypes;
using System.Collections.Generic;
using System.Reflection;

namespace Unit_Tests
{
    public class KernelTests
    {
        public KernelTests()
        {

            ctx = new CudaContext(); // we must call this first
            int Len = 100000;
            var A = new SyncVariable<float3>(GenRandomVectors(Len));
            var B = new SyncVariable<float3>(GenRandomVectors(Len));
            var C = new SyncVariable<float3>(Len);
            var D = new SyncVariable<float>(Len);
            var Length = new SyncVariable<int>(new int[]{Len}); // instead of an int use an int array of length 1
            var TrisCpu = new FEA.Mesher.TriangleSTL[Len];
            var rng = new Random();
            for (int i = 0; i < Len; i++)
            {
                TrisCpu[i].Vertex1.x = (float)rng.NextDouble();
                TrisCpu[i].Vertex1.y = (float)rng.NextDouble();
                TrisCpu[i].Vertex1.z = (float)rng.NextDouble();

                TrisCpu[i].Vertex2.x = (float)rng.NextDouble();
                TrisCpu[i].Vertex2.y = (float)rng.NextDouble();
                TrisCpu[i].Vertex2.z = (float)rng.NextDouble();

                TrisCpu[i].Vertex3.x = (float)rng.NextDouble();
                TrisCpu[i].Vertex3.y = (float)rng.NextDouble();
                TrisCpu[i].Vertex3.z = (float)rng.NextDouble();
            }
            var Tris = new SyncVariable<FEA.Mesher.TriangleSTL>(TrisCpu);


            var PtxFile = "KernelUnitTests.ptx";
            var CrossProdKernel = ctx.LoadKernelPTX(PtxFile, "TestCrossProduct"); 
            var AddKernel = ctx.LoadKernelPTX(PtxFile, "TestAdd"); 
            var SubKernel = ctx.LoadKernelPTX(PtxFile, "TestSubtract");
            var DotKernel = ctx.LoadKernelPTX(PtxFile, "TestDotProduct");
            var AreaKernel = ctx.LoadKernelPTX(PtxFile, "TestTriangleArea");
            var IntersectionKernel = ctx.LoadKernelPTX(PtxFile, "TestPlaneIntersection");
            var BlockDims = new dim3(512);
            var GridDims = new dim3(Len / 512 + 1);

            CrossProdKernel.BlockDimensions = BlockDims;
            CrossProdKernel.GridDimensions = GridDims;
            AddKernel.BlockDimensions = BlockDims;
            AddKernel.GridDimensions = GridDims;
            SubKernel.BlockDimensions = BlockDims;
            SubKernel.GridDimensions = GridDims;
            DotKernel.BlockDimensions = BlockDims;
            DotKernel.GridDimensions = GridDims;
            AreaKernel.BlockDimensions = BlockDims;
            AreaKernel.GridDimensions = GridDims;

            CrossProdKernel.Run(Len, A.GPUPtr(), B.GPUPtr(), C.GPUPtr());
            A.Sync();
            B.Sync();
            C.Sync();
            float eps = 1e-7f;
            for (int i = 0; i < Len; i++)
            {
                var Ans = A.cpuArray[i].Cross(B.cpuArray[i]) - C.cpuArray[i];

                if (Ans.Length >= eps)
                {
                    throw new Exception("Test Failed");
                }
            }
            AddKernel.Run(Len, A.GPUPtr(), B.GPUPtr(), C.GPUPtr());
            A.Sync();
            B.Sync();
            C.Sync();
            for (int i = 0; i < Len; i++)
            {
                var Ans = A.cpuArray[i] + B.cpuArray[i] - C.cpuArray[i];

                if (Ans.Length >= eps)
                {
                    throw new Exception("Test Failed");
                }
            }
            SubKernel.Run(Len, A.GPUPtr(), B.GPUPtr(), C.GPUPtr());
            A.Sync();
            B.Sync();
            C.Sync();
            for (int i = 0; i < Len; i++)
            {
                var Ans = A.cpuArray[i] - B.cpuArray[i] - C.cpuArray[i];

                if (Ans.Length >= eps)
                {
                    throw new Exception("Test Failed");
                }
            }
            DotKernel.Run(Len, A.GPUPtr(), B.GPUPtr(), D.GPUPtr());
            A.Sync();
            B.Sync();
            D.Sync();
            for (int i = 0; i < Len; i++)
            {
                float Ans = A.cpuArray[i].Dot(B.cpuArray[i]) - D.cpuArray[i];

                if (Ans >= 3*eps)
                {
                    throw new Exception("Test Failed");
                }
            }
            AreaKernel.Run(Len, Tris.GPUPtr(), D.GPUPtr());
            Tris.Sync();
            D.Sync();
            for (int i = 0; i < Len; i++)
            {
                float ans = D.cpuArray[i];
            }

        }
        //public ExecuteKernel()

        public static float3[] GenRandomVectors(int Len) {
            var rng = new Random();
            var Ans = new float3[Len];
            for (int i = 0; i < Len; i++)
            {
                Ans[i] = new float3();
                Ans[i].x = (float)rng.NextDouble();
                Ans[i].y = (float)rng.NextDouble();
                Ans[i].z = (float)rng.NextDouble();
            }
            return Ans;
        }
        private Dictionary<Type, string> Typemap; // this is for type and argument checking of the kernel
        private Dictionary<string, CudaKernel> Kernels;
        private Dictionary<string, Type> VariableTypes;
        private Dictionary<string, CudaDeviceVariable<Byte>> CudaVariables; // pass either device pointers or integers
        private CudaContext ctx;
        private string PtxFile;

        public void addKernel(string KernelName, CudaKernel Kernel) {
            Kernels.Add(KernelName, Kernel);
        }
        public void addVariable(string VariableName, CudaDeviceVariable<Byte> Value, Type VariableType) {
            CudaVariables.Add(VariableName, Value);
            VariableTypes.Add(VariableName, VariableType);
        } 

        public void TestOutput(Byte[] Ans, Byte[] ExpectedValue) {
            for (int i = 0; i < Ans.Length; i++)
            {
                if (Ans[i] != ExpectedValue[i])
                {
                    throw new Exception("Value not Valid");
                }
            }
        }

    }

    public class SyncVariable<T> : IDisposable
    {
        public CudaDeviceVariable<Byte> gpuArray;
        public T[] cpuArray;

        public SyncVariable(int Len) {
            cpuArray = new T[Len];
            gpuArray = new CudaDeviceVariable<Byte>(Len * sizeof(T));
        } 
        public SyncVariable(T[] Variable) {
            cpuArray = Variable;
            gpuArray = new CudaDeviceVariable<Byte>(cpuArray.Length * sizeof(T));
            unsafe {
                fixed (T* p = Variable)
                {
                    IntPtr ptr = (IntPtr) p;
                    gpuArray.CopyToDevice(ptr);
                }   
            }
        } 
        public void Sync() {
            cpuArray = gpuArray;
        }
        public CUdeviceptr GPUPtr() {
            return gpuArray.DevicePointer;
        }

        public void Dispose() { // free memory allocated on gpu
            gpuArray.Dispose();
        }

    }



}

