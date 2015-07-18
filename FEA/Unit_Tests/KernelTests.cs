using System;
using ManagedCuda;
using ManagedCuda.VectorTypes;
using ManagedCuda.BasicTypes;

namespace Unit_Tests
{
    public class KernelTests
    {
        public KernelTests()
        {
            var ctx = new CudaContext(); // we must call this first
            int Len = 100000;
            var A = new SyncVariable<float3>(GenRandomVectors(Len));
            var B = new SyncVariable<float3>(GenRandomVectors(Len));
            var C = new SyncVariable<float3>(Len);

            var CrossProdKernel = ctx.LoadKernelPTX("KernelUnitTests.ptx", "TestCrossProduct"); 
            //var AddKernel = ctx.LoadKernelPTX("KernelUnitTests.ptx", "TestAd"); 
            CrossProdKernel.BlockDimensions.x = 8;
            CrossProdKernel.BlockDimensions.y = 8;
            CrossProdKernel.BlockDimensions.z = 8;
            CrossProdKernel.GridDimensions = Len / 512 + 1;
            CrossProdKernel.Run(Len, A.GPUPtr(), B.GPUPtr(), C.GPUPtr());
            A.Sync();
            B.Sync();
            C.Sync();
            for (int i = 0; i < Len; i++)
            {
                var Ans = A.cpuArray[i].Cross(B.cpuArray[i]) - C.cpuArray[i];
                if (Ans.Length != 0)
                {
                    throw new Exception("Test Failed");
                }
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
        }

    }
        

    public class SyncVariable<T>
    where T : struct
    {
        public CudaDeviceVariable<T> gpuArray;
        public T[] cpuArray;

        public SyncVariable(int Len) {
            cpuArray = new T[Len];
            gpuArray = new CudaDeviceVariable<T>(Len);
        } 
        public SyncVariable(T[] Variable) {
            cpuArray = Variable;
            gpuArray = new CudaDeviceVariable<T>(cpuArray.Length);
            gpuArray = Variable;
        } 
        public void Sync() {
            cpuArray = gpuArray;
        }
        public CUdeviceptr GPUPtr() {
            return gpuArray.DevicePointer;
        }
    }
}

