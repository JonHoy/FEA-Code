using System;
using System.Collections.Generic;
using System.Text;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace ManagedCuda.Dispatched
{
    /// <summary>
    /// A CUDA function or CUDA kernel
    /// </summary>
    [Obsolete("CUDA 4.0 API calls are now thread safe. Dispatched... wrapper classes are of no use now.")]
    public class DispatchedCudaKernel : DispatchedCudaBaseClass, IDisposable
    {
        private DispatchedCudaContext _cuda;
        private CUmodule _module;
        private CUfunction _function;

        private uint _sharedMemSize;
        private int _paramOffset;
        private dim3 _blockDim;
        private dim3 _gridDim;
        private string _kernelName;
        private CUResult res;
        private bool disposed;

        #region Constructors
        /// <summary>
        /// Loads the given CUDA kernel from the CUmodule. Block and Grid dimensions must be set 
        /// before running the kernel. Shared memory size is set to 0.
        /// </summary>
        /// <param name="kernelName">The kernel name as defined in the *.cu file</param>
        /// <param name="module">The CUmodule which contains the kernel</param>
        /// <param name="cuda">CUDA abstraction layer object (= CUDA context) for this Kernel</param>
        public DispatchedCudaKernel(string kernelName, CUmodule module, DispatchedCudaContext cuda)
            : base(cuda.Dispatcher)
        {  
            _module = module;
            _cuda = cuda;
            _kernelName = kernelName;

            object[] paramModuleGetFunction = { _function, _module, _kernelName };

            res = (CUResult)_dispatcher.Invoke(new cuModuleGetFunction(DriverAPINativeMethods.ModuleManagement.cuModuleGetFunction), paramModuleGetFunction);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetFunction", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetFunction(ref _function, _module, _kernelName);
            if (res != CUResult.Success) throw new CudaException(res);

            _function = (CUfunction)paramModuleGetFunction[0];

            _blockDim.x = _blockDim.y = 32;        
            _blockDim.z = 0;
            _gridDim.x = _gridDim.y = _gridDim.z = 1;
        }

        /// <summary>
        /// Loads the given CUDA kernel from the CUmodule. Block and Grid dimensions are set directly. 
        /// Shared memory size is set to 0.
        /// </summary>
        /// <param name="kernelName">The kernel name as defined in the *.cu file</param>
        /// <param name="module">The CUmodule which contains the kernel</param>
        /// <param name="cuda">CUDA abstraction layer object (= CUDA context) for this Kernel</param>
        /// <param name="blockDim">Dimension of block of threads (3D)</param>
        /// <param name="gridDim">Dimension of grid of blocks of threads (2D - z-component is discarded)</param>
        public DispatchedCudaKernel(string kernelName, CUmodule module, DispatchedCudaContext cuda, dim3 blockDim, dim3 gridDim)
            : this(kernelName, module, cuda)
        {
            _blockDim = blockDim;
            _gridDim = gridDim;
            _gridDim.z = 1;
        }

        /// <summary>
        /// Loads the given CUDA kernel from the CUmodule. Block dimensions are set directly, 
        /// grid dimensions must be set before running the kernel. Shared memory size is set to 0.
        /// </summary>
        /// <param name="kernelName">The kernel name as defined in the *.cu file</param>
        /// <param name="module">The CUmodule which contains the kernel</param>
        /// <param name="cuda">CUDA abstraction layer object (= CUDA context) for this Kernel</param>
        /// <param name="blockDim">Dimension of block of threads</param>
        public DispatchedCudaKernel(string kernelName, CUmodule module, DispatchedCudaContext cuda, dim3 blockDim)
            : this(kernelName, module, cuda)
        {
            _blockDim = blockDim;
        }

        /// <summary>
        /// Loads the given CUDA kernel from the CUmodule. Block and Grid dimensions are set directly. 
        /// Shared memory size is set to 0.
        /// </summary>
        /// <param name="kernelName">The kernel name as defined in the *.cu file</param>
        /// <param name="module">The CUmodule which contains the kernel</param>
        /// <param name="cuda">CUDA abstraction layer object (= CUDA context) for this Kernel</param>
        /// <param name="blockDimX">Dimension of block of threads X</param>
        /// <param name="blockDimY">Dimension of block of threads Y</param>
        /// <param name="blockDimZ">Dimension of block of threads Z</param>
        /// <param name="gridDimX">Dimension of grid of block of threads X</param>
        /// <param name="gridDimY">Dimension of grid of block of threads Y</param>
        public DispatchedCudaKernel(string kernelName, CUmodule module, DispatchedCudaContext cuda, uint blockDimX, uint blockDimY, uint blockDimZ, uint gridDimX, uint gridDimY)
            : this(kernelName, module, cuda)
        {
            _blockDim.x = blockDimX;
            _blockDim.y = blockDimY;
            _blockDim.z = blockDimZ;
            _gridDim.x = gridDimX;
            _gridDim.y = gridDimY;
            _gridDim.z = 1;
        }

        /// <summary>
        /// Loads the given CUDA kernel from the CUmodule. Block dimensions are set directly, 
        /// grid dimensions must be set before running the kernel. Shared memory size is set to 0.
        /// </summary>
        /// <param name="kernelName">The kernel name as defined in the *.cu file</param>
        /// <param name="module">The CUmodule which contains the kernel</param>
        /// <param name="cuda">CUDA abstraction layer object (= CUDA context) for this Kernel</param>
        /// <param name="blockDimX">Dimension of block of threads X</param>
        /// <param name="blockDimY">Dimension of block of threads Y</param>
        /// <param name="blockDimZ">Dimension of block of threads Z</param>
        public DispatchedCudaKernel(string kernelName, CUmodule module, DispatchedCudaContext cuda, uint blockDimX, uint blockDimY, uint blockDimZ)
            : this(kernelName, module, cuda)
        {
            _blockDim.x = blockDimX;
            _blockDim.y = blockDimY;
            _blockDim.z = blockDimZ;
        }

        /// <summary>
        /// Loads the given CUDA kernel from the CUmodule. Block and Grid dimensions must be set before running the kernel. 
        /// Shared memory size is set directly.
        /// </summary>
        /// <param name="kernelName">The kernel name as defined in the *.cu file</param>
        /// <param name="module">The CUmodule which contains the kernel</param>
        /// <param name="cuda">CUDA abstraction layer object (= CUDA context) for this Kernel</param>
        /// <param name="sharedMemory">Dynamic shared memory size in Bytes</param>
        public DispatchedCudaKernel(string kernelName, CUmodule module, DispatchedCudaContext cuda, uint sharedMemory)
            : this(kernelName, module, cuda)
        {
            _sharedMemSize = sharedMemory;
        }

        /// <summary>
        /// Loads the given CUDA kernel from the CUmodule. Block and Grid dimensions and shared memory size are set directly.
        /// </summary>
        /// <param name="kernelName">The kernel name as defined in the *.cu file</param>
        /// <param name="module">The CUmodule which contains the kernel</param>
        /// <param name="cuda">CUDA abstraction layer object (= CUDA context) for this Kernel</param>
        /// <param name="blockDim">Dimension of block of threads (3D)</param>
        /// <param name="gridDim">Dimension of grid of blocks of threads (2D - z-component is discarded)</param>
        /// <param name="sharedMemory">Dynamic shared memory size in Bytes</param>
        public DispatchedCudaKernel(string kernelName, CUmodule module, DispatchedCudaContext cuda, dim3 blockDim, dim3 gridDim, uint sharedMemory)
            : this(kernelName, module, cuda, blockDim, gridDim)
        {
            _sharedMemSize = sharedMemory;
        }

        /// <summary>
        /// Loads the given CUDA kernel from the CUmodule. Block dimensions and shared memors size are set directly, 
        /// grid dimensions must be set before running the kernel.
        /// </summary>
        /// <param name="kernelName">The kernel name as defined in the *.cu file</param>
        /// <param name="module">The CUmodule which contains the kernel</param>
        /// <param name="cuda">CUDA abstraction layer object (= CUDA context) for this Kernel</param>
        /// <param name="blockDim">Dimension of block of threads</param>
        /// <param name="sharedMemory">Dynamic shared memory size in Bytes</param>
        public DispatchedCudaKernel(string kernelName, CUmodule module, DispatchedCudaContext cuda, dim3 blockDim, uint sharedMemory)
            : this(kernelName, module, cuda, blockDim)
        {
            _sharedMemSize = sharedMemory;
        }

        /// <summary>
        /// Loads the given CUDA kernel from the CUmodule. Block dimensions and shared memors size are set directly, 
        /// grid dimensions must be set before running the kernel.
        /// </summary>
        /// <param name="kernelName">The kernel name as defined in the *.cu file</param>
        /// <param name="module">The CUmodule which contains the kernel</param>
        /// <param name="cuda">CUDA abstraction layer object (= CUDA context) for this Kernel</param>
        /// <param name="blockDimX">Dimension of block of threads X</param>
        /// <param name="blockDimY">Dimension of block of threads Y</param>
        /// <param name="blockDimZ">Dimension of block of threads Z</param>
        /// <param name="sharedMemory">Dynamic shared memory size in Bytes</param>
        public DispatchedCudaKernel(string kernelName, CUmodule module, DispatchedCudaContext cuda, uint blockDimX, uint blockDimY, uint blockDimZ, uint sharedMemory)
            : this(kernelName, module, cuda, blockDimX, blockDimY, blockDimZ)
        {
            _sharedMemSize = sharedMemory;
        }

        /// <summary>
        /// 
        /// </summary>
        ~DispatchedCudaKernel()
        {
            Dispose (false);
        }
        #endregion

        #region Dispose
        /// <summary>
        /// 
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="fDisposing"></param>
        protected virtual void Dispose (bool fDisposing)
        {
            if (fDisposing && !disposed) 
            {
                _cuda.UnloadModule(_module);
                disposed = true;
            }
        }
        #endregion

        #region setConstantVaiable
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, byte value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDByte(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, sbyte value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDSByte(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, ushort value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUShort(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, short value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDShort(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, uint value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUInt(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, int value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDInt(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, ulong value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDULong(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, long value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDLong(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, float value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloat(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, double value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDDouble(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        #region VectorTypes

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, dim3 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDDim3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, char1 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDChar1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, char2 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDChar2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, char3 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDChar3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, char4 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDChar4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, uchar1 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUChar1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, uchar2 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUChar2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, uchar3 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUChar3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, uchar4 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUChar4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, short1 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDShort1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, short2 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDShort2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, short3 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDShort3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, short4 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDShort4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, ushort1 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUShort1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, ushort2 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUShort2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, ushort3 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUShort3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, ushort4 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUShort4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, int1 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDInt1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, int2 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDInt2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, int3 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDInt3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, int4 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDInt4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, uint1 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUInt1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, uint2 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUInt2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, uint3 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUInt3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, uint4 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUInt4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, long1 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDLong1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, long2 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDLong2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, long3 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDLong3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, long4 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDLong4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, ulong1 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDULong1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, ulong2 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDULong2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, ulong3 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDULong3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, ulong4 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDULong4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, float1 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloat1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, float2 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloat2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, float3 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloat3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, float4 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloat4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, double1 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDDouble1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, double2 value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDDouble2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, cuDoubleComplex value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDDoubleComplex(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, cuDoubleReal value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDDoubleReal(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, cuFloatComplex value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloatComplex(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, cuFloatReal value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloatReal(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        #endregion
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, byte[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDByteA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, sbyte[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDSByteA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, ushort[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUShortA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, short[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDShortA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, uint[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUIntA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, int[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDIntA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, ulong[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDULongA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, long[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDLongA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, float[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloatA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, double[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDDoubleA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        #region VectorTypes
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, dim3[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDDim3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, char1[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDChar1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, char2[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDChar2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, char3[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDChar3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, char4[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDChar4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, uchar1[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUChar1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, uchar2[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUChar2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, uchar3[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUChar3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, uchar4[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUChar4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, short1[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDShort1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, short2[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDShort2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, short3[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDShort3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, short4[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDShort4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, ushort1[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUShort1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, ushort2[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUShort2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, ushort3[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUShort3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, ushort4[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUShort4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, int1[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDInt1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, int2[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDInt2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, int3[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDInt3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, int4[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDInt4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, uint1[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUInt1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, uint2[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUInt2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, uint3[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUInt3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, uint4[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUInt4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, long1[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDLong1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, long2[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDLong2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, long3[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDLong3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, long4[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDLong4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, ulong1[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDULong1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, ulong2[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDULong2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, ulong3[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDULong3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, ulong4[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDULong4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, float1[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloat1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, float2[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloat2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, float3[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloat3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, float4[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloat4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, double1[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDDouble1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, double2[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDDouble2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, cuDoubleComplex[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDDoubleComplexA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, cuDoubleReal[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDDoubleRealA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, cuFloatComplex[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloatComplexA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        /// <summary>
        /// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
        /// </summary>
        /// <param name="name">constant variable name</param>
        /// <param name="value">value</param>
        public void SetConstantVariable(string name, cuFloatReal[] value)
        {
            CUdeviceptr dVarPtr = new CUdeviceptr();
            SizeT varSize = 0;

            object[] paramModuleGetGlobal = { dVarPtr, varSize, _module, name };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetGlobal_v2(DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2), paramModuleGetGlobal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleGetGlobal", res));
            //res = DriverAPI.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
            if (res != CUResult.Success) throw new CudaException(res);
            dVarPtr = (CUdeviceptr)paramModuleGetGlobal[0];
            varSize = (uint)paramModuleGetGlobal[1];

            object[] paramMemcpyHtoD = { dVarPtr, value, varSize };
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloatRealA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), paramMemcpyHtoD);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        #endregion
        #endregion

        #region SetParameter
        /// <summary>
        /// Computes the alignment for almost all datatypes
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        private static int AlignOf(object type)
        {
            if (type is byte || type is sbyte)
                return 1;
            if (type is short || type is ushort)
                return 2;
            if (type is int || type is uint || type is float)
                return 4;
            if (type is long || type is ulong || type is double)
                return 8;
            
            if (type is char1) return 1;
            if (type is char2) return 2;
            if (type is char3) return 1;
            if (type is char4) return 4;
            if (type is uchar1) return 1;
            if (type is uchar2) return 2;
            if (type is uchar3) return 1;
            if (type is uchar4) return 4;

            if (type is short1) return 2;
            if (type is short2) return 4;
            if (type is short3) return 2;
            if (type is short4) return 8;
            if (type is ushort1) return 2;
            if (type is ushort2) return 4;
            if (type is ushort3) return 2;
            if (type is ushort4) return 8;

            if (type is int1) return 4;
            if (type is int2) return 8;
            if (type is int3) return 4;
            if (type is int4) return 16;
            if (type is uint1) return 4;
            if (type is uint2) return 8;
            if (type is uint3) return 4;
            if (type is uint4) return 16;

            if (type is long1) return 4;
            if (type is long2) return 8;
            if (type is long3) return 4;
            if (type is long4) return 16;
            if (type is ulong1) return 4;
            if (type is ulong2) return 8;
            if (type is ulong3) return 4;
            if (type is ulong4) return 16;

            if (type is float1) return 4;
            if (type is float2) return 8;
            if (type is float3) return 4;
            if (type is float4) return 16;

            if (type is double1) return 8;
            if (type is double2) return 16;
            if (type is cuFloatComplex) return 8;
            if (type is cuDoubleComplex) return 16;
            if (type is cuFloatReal) return 4;
            if (type is cuDoubleReal) return 8;

            if (type is CUdeviceptr) return IntPtr.Size;
            if (type is IntPtr) return IntPtr.Size;

            throw new CudaException(CUResult.ErrorInvalidValue);
        }

        /// <summary>
        /// Set a value as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(byte value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvByte(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }

        /// <summary>
        /// Set a value as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(sbyte value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvSByte(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }

        /// <summary>
        /// Set a value as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(short value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvShort(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }

        /// <summary>
        /// Set a value as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(ushort value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvUShort(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }

        /// <summary>
        /// Set a value as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(int value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            //object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSeti(DriverAPINativeMethods.ParameterManagement.cuParamSeti), _function, _paramOffset, (uint)value);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSeti", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }

        /// <summary>
        /// Set a value as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(uint value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            //object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSeti(DriverAPINativeMethods.ParameterManagement.cuParamSeti), _function, _paramOffset, value);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSeti", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }

        /// <summary>
        /// Set a value as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(ulong value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvULong(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }

        /// <summary>
        /// Set a value as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(long value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvLong(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }

        /// <summary>
        /// Set a value as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(float value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetF = { _function, _paramOffset, value };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetf(DriverAPINativeMethods.ParameterManagement.cuParamSetf), paramParamSetF);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetf", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetf(_function, _paramOffset, value);
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }

        /// <summary>
        /// Set a value as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(double value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvDouble(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }

        /// <summary>
        /// Set a value as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(SizeT value)
        {
            if (IntPtr.Size == 8)
            {
                ulong temp = value;
                SetParameter(temp);
            }
            else
            {
                uint temp = value;
                SetParameter(temp);
            }
        }

        #region VectorTypes
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(dim3 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvDim3(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }

        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(char1 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvChar1(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(char2 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvChar2(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(char3 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvChar3(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(char4 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvChar4(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }

        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(uchar1 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvUChar1(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(uchar2 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvUChar2(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(uchar3 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvUChar3(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(uchar4 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvUChar4(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }

        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(short1 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvShort1(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(short2 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvShort2(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(short3 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvShort3(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(short4 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvShort4(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }

        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(ushort1 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvUShort1(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(ushort2 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvUShort2(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(ushort3 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvUShort3(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(ushort4 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvUShort4(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }

        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(int1 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvInt1(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(int2 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvInt2(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(int3 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvInt3(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(int4 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvInt4(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }

        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(uint1 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvUInt1(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(uint2 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvUInt2(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(uint3 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvUInt3(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(uint4 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvUInt4(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }

        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(long1 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvLong1(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(long2 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvLong2(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(long3 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvLong3(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(long4 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvLong4(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }

        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(ulong1 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvULong1(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(ulong2 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvULong2(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(ulong3 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvULong3(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(ulong4 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvULong4(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }

        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(float1 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvFloat1(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(float2 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvFloat2(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(float3 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvFloat3(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(float4 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvFloat4(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }

        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(double1 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvDouble1(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(double2 value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvDouble2(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(cuDoubleComplex value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvDoubleComplex(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(cuDoubleReal value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvDoubleReal(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(cuFloatComplex value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvFloatComplex(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        /// <summary>
        /// Set a Cuda Vector Type as a kernel parameter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(cuFloatReal value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetV = { _function, _paramOffset, value, (uint)Marshal.SizeOf(value) };
            res = (CUResult)_dispatcher.Invoke(new cuParamSetvFloatReal(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, ref value, (uint)Marshal.SizeOf(value));
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += Marshal.SizeOf(value);
        }
        #endregion

        /// <summary>
        /// Set a CUdeviceptr as a kernel paramter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter(CUdeviceptr value)
        {
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

            object[] paramParamSetI = { _function, _paramOffset, value.Pointer };
            res = (CUResult)_dispatcher.Invoke(new cuParamSeti(DriverAPINativeMethods.ParameterManagement.cuParamSeti), paramParamSetI);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSeti", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSeti(_function, _paramOffset, value.Pointer);
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += IntPtr.Size;
        }

        ///// <summary>
        ///// Set an IntPtr as a kernel paramter
        ///// </summary>
        ///// <param name="value">parameter value</param>
        //public void SetParameter(IntPtr value)
        //{
        //    _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);

        //    object[] paramParamSetV = { _function, _paramOffset, value, (uint)IntPtr.Size };
        //    res = (CUResult)_dispatcher.Invoke(new cuParamSetvIntPtr(DriverAPINativeMethods.ParameterManagement.cuParamSetv), paramParamSetV);
        //    Debug.WriteLine("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSetv", res, _paramOffset);   
        //    //res = DriverAPI.ParameterManagement.cuParamSetv(_function, _paramOffset, value, (uint)IntPtr.Size);
        //    if (res != CUResult.Success) throw new CudaException(res);
        //    _paramOffset += Marshal.SizeOf(value);
        //}

        /// <summary>
        /// Set a CudaDeviceVariable as a kernel paramter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter<T>(DispatchedCudaDeviceVariable<T> value) where T : struct
        {
            if (disposed) throw new ObjectDisposedException(_kernelName);
            CUdeviceptr ptr = value.DevicePointer;
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);
            object[] paramParamSetI = { _function, _paramOffset, ptr.Pointer };
            res = (CUResult)_dispatcher.Invoke(new cuParamSeti(DriverAPINativeMethods.ParameterManagement.cuParamSeti), paramParamSetI);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSeti", res, _paramOffset));
            //res = DriverAPI.ParameterManagement.cuParamSeti(_function, _paramOffset, value.Pointer);
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += IntPtr.Size;
        }

        /// <summary>
        /// Set a CudaPitchedDeviceVariable as a kernel paramter
        /// </summary>
        /// <param name="value">parameter value</param>
        public void SetParameter<T>(DispatchedCudaPitchedDeviceVariable<T> value) where T : struct
        {
            if (disposed) throw new ObjectDisposedException(_kernelName);
            CUdeviceptr ptr = value.DevicePointer;
            _paramOffset = (_paramOffset + AlignOf(value) - 1) & ~(AlignOf(value) - 1);
            object[] paramParamSetI = { _function, _paramOffset, ptr.Pointer };
            res = (CUResult)_dispatcher.Invoke(new cuParamSeti(DriverAPINativeMethods.ParameterManagement.cuParamSeti), paramParamSetI);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter offset: {3}", DateTime.Now, "cuParamSeti", res, _paramOffset));
            //res = DriverAPI.ParameterManagement.cuParamSeti(_function, _paramOffset, value.Pointer);
            if (res != CUResult.Success) throw new CudaException(res);
            _paramOffset += IntPtr.Size;
        }

        private void setParameterObject(object value)
        {
            if (value is byte)
            {
                SetParameter((byte)value);
                return;
            }
            if (value is sbyte)
            {
                 SetParameter((sbyte)value);
                return;
            }
            if (value is short)
            {
                 SetParameter((short)value);
                return;
            }
            if (value is ushort) 
            {
                SetParameter((ushort)value);
                return;
            }
            if (value is int)
            {
                 SetParameter((int)value);
                return;
            }
            if (value is uint) 
            {
                SetParameter((uint)value);
                return;
            }
            if (value is long) 
            {
                SetParameter((long)value);
                return;
            }
            if (value is ulong) 
            {
                SetParameter((ulong)value);
                return;
            }
            if (value is float) 
            {
                SetParameter((float)value);
                return;
            }
            if (value is double)
            {
                 SetParameter((double)value);
                return;
            }
            if (value is SizeT)
            {
                if (IntPtr.Size == 8)
                {
                    ulong temp = (SizeT)value;
                    SetParameter(temp);
                }
                else
                {
                    uint temp = (SizeT)value;
                    SetParameter(temp);
                }
                return;
            }

            //if (value is byte[]) 
            //{
            //    SetParameter((byte[])value);
            //    return;
            //}
            //if (value is sbyte[])
            //{
            //     SetParameter((sbyte[])value);
            //    return;
            //}
            //if (value is short[]) 
            //{
            //    SetParameter((short[])value);
            //    return;
            //}
            //if (value is ushort[])
            //{
            //     SetParameter((ushort[])value);
            //    return;
            //}
            //if (value is int[]) 
            //{
            //    SetParameter((int[])value);
            //    return;
            //}
            //if (value is uint[])
            //{
            //     SetParameter((uint[])value);
            //    return;
            //}
            //if (value is long[])
            //{
            //     SetParameter((long[])value);
            //    return;
            //}
            //if (value is ulong[]) 
            //{
            //    SetParameter((ulong[])value);
            //    return;
            //}
            //if (value is float[])
            //{
            //     SetParameter((float[])value);
            //    return;
            //}
            //if (value is double[])
            //{
            //     SetParameter((double[])value);
            //    return;
            //}

            if (value is dim3)
            {
                SetParameter((dim3)value);
                return;
            }
            if (value is char1)
            {
                SetParameter((char1)value);
                return;
            }
            if (value is char2)
            {
                SetParameter((char2)value);
                return;
            }
            if (value is char3)
            {
                SetParameter((char3)value);
                return;
            }
            if (value is char4)
            {
                SetParameter((char4)value);
                return;
            }
            if (value is uchar1)
            {
                SetParameter((uchar1)value);
                return;
            }
            if (value is uchar2)
            {
                SetParameter((uchar2)value);
                return;
            }
            if (value is uchar3)
            {
                SetParameter((uchar3)value);
                return;
            }
            if (value is uchar4)
            {
                SetParameter((uchar4)value);
                return;
            }

            if (value is short1)
            {
                SetParameter((short1)value);
                return;
            }
            if (value is short2)
            {
                SetParameter((short2)value);
                return;
            }
            if (value is short3)
            {
                SetParameter((short3)value);
                return;
            }
            if (value is short4)
            {
                SetParameter((short4)value);
                return;
            }
            if (value is ushort1)
            {
                SetParameter((ushort1)value);
                return;
            }
            if (value is ushort2)
            {
                SetParameter((ushort2)value);
                return;
            }
            if (value is ushort3)
            {
                SetParameter((ushort3)value);
                return;
            }
            if (value is ushort4)
            {
                SetParameter((ushort4)value);
                return;
            }

            if (value is int1)
            {
                SetParameter((int1)value);
                return;
            }
            if (value is int2)
            {
                SetParameter((int2)value);
                return;
            }
            if (value is int3)
            {
                SetParameter((int3)value);
                return;
            }
            if (value is int4)
            {
                SetParameter((int4)value);
                return;
            }
            if (value is uint1)
            {
                SetParameter((uint1)value);
                return;
            }
            if (value is uint2)
            {
                SetParameter((uint2)value);
                return;
            }
            if (value is uint3)
            {
                SetParameter((uint3)value);
                return;
            }
            if (value is uint4)
            {
                SetParameter((uint4)value);
                return;
            }

            if (value is long1)
            {
                SetParameter((long1)value);
                return;
            }
            if (value is long2)
            {
                SetParameter((long2)value);
                return;
            }
            if (value is long3)
            {
                SetParameter((long3)value);
                return;
            }
            if (value is long4)
            {
                SetParameter((long4)value);
                return;
            }
            if (value is ulong1)
            {
                SetParameter((ulong1)value);
                return;
            }
            if (value is ulong2)
            {
                SetParameter((ulong2)value);
                return;
            }
            if (value is ulong3)
            {
                SetParameter((ulong3)value);
                return;
            }
            if (value is ulong4)
            {
                SetParameter((ulong4)value);
                return;
            }

            if (value is float1)
            {
                SetParameter((float1)value);
                return;
            }
            if (value is float2)
            {
                SetParameter((float2)value);
                return;
            }
            if (value is float3)
            {
                SetParameter((float3)value);
                return;
            }
            if (value is float4)
            {
                SetParameter((float4)value);
                return;
            }
            if (value is double1)
            {
                SetParameter((double1)value);
                return;
            }
            if (value is double2)
            {
                SetParameter((double2)value);
                return;
            }
            if (value is cuDoubleComplex)
            {
                SetParameter((cuDoubleComplex)value);
                return;
            }
            if (value is cuDoubleReal)
            {
                SetParameter((cuDoubleReal)value);
                return;
            }
            if (value is cuFloatComplex)
            {
                SetParameter((cuFloatComplex)value);
                return;
            }
            if (value is cuFloatReal)
            {
                SetParameter((cuFloatReal)value);
                return;
            }
            #region comments
            /*/if (value is dim1[])
            //{
            //    SetParameter((dim1[])value);
            //    return;
            //}
            //if (value is dim2[])
            //{
            //    SetParameter((dim2[])value);
            //    return;
            //}
            //if (value is dim3[])
            //{
            //    SetParameter((dim3[])value);
            //    return;
            //}
            //if (value is dim4[])
            //{
            //    SetParameter((dim4[])value);
            //    return;
            //}
            //if (value is char1[])
            //{
            //    SetParameter((char1[])value);
            //    return;
            //}
            //if (value is char2[])
            //{
            //    SetParameter((char2[])value);
            //    return;
            //}
            //if (value is char3[])
            //{
            //    SetParameter((char3[])value);
            //    return;
            //}
            //if (value is char4[])
            //{
            //    SetParameter((char4[])value);
            //    return;
            //}
            //if (value is uchar1[])
            //{
            //    SetParameter((uchar1[])value);
            //    return;
            //}
            //if (value is uchar2[])
            //{
            //    SetParameter((uchar2[])value);
            //    return;
            //}
            //if (value is uchar3[])
            //{
            //    SetParameter((uchar3[])value);
            //    return;
            //}
            //if (value is uchar4[])
            //{
            //    SetParameter((uchar4[])value);
            //    return;
            //}

            //if (value is short1[])
            //{
            //    SetParameter((short1[])value);
            //    return;
            //}
            //if (value is short2[])
            //{
            //    SetParameter((short2[])value);
            //    return;
            //}
            //if (value is short3[])
            //{
            //    SetParameter((short3[])value);
            //    return;
            //}
            //if (value is short4[])
            //{
            //    SetParameter((short4[])value);
            //    return;
            //}
            //if (value is ushort1[])
            //{
            //    SetParameter((ushort1[])value);
            //    return;
            //}
            //if (value is ushort2[])
            //{
            //    SetParameter((ushort2[])value);
            //    return;
            //}
            //if (value is ushort3[])
            //{
            //    SetParameter((ushort3[])value);
            //    return;
            //}
            //if (value is ushort4[])
            //{
            //    SetParameter((ushort4[])value);
            //    return;
            //}

            //if (value is int1[])
            //{
            //    SetParameter((int1[])value);
            //    return;
            //}
            //if (value is int2[])
            //{
            //    SetParameter((int2[])value);
            //    return;
            //}
            //if (value is int3[])
            //{
            //    SetParameter((int3[])value);
            //    return;
            //}
            //if (value is int4[])
            //{
            //    SetParameter((int4[])value);
            //    return;
            //}
            //if (value is uint1[])
            //{
            //    SetParameter((uint1[])value);
            //    return;
            //}
            //if (value is uint2[])
            //{
            //    SetParameter((uint2[])value);
            //    return;
            //}
            //if (value is uint3[])
            //{
            //    SetParameter((uint3[])value);
            //    return;
            //}
            //if (value is uint4[])
            //{
            //    SetParameter((uint4[])value);
            //    return;
            //}

            //if (value is long1[])
            //{
            //    SetParameter((long1[])value);
            //    return;
            //}
            //if (value is long2[])
            //{
            //    SetParameter((long2[])value);
            //    return;
            //}
            //if (value is long3[])
            //{
            //    SetParameter((long3[])value);
            //    return;
            //}
            //if (value is long4[])
            //{
            //    SetParameter((long4[])value);
            //    return;
            //}
            //if (value is ulong1[])
            //{
            //    SetParameter((ulong1[])value);
            //    return;
            //}
            //if (value is ulong2[])
            //{
            //    SetParameter((ulong2[])value);
            //    return;
            //}
            //if (value is ulong3[])
            //{
            //    SetParameter((ulong3[])value);
            //    return;
            //}
            //if (value is ulong4[])
            //{
            //    SetParameter((ulong4[])value);
            //    return;
            //}

            //if (value is float1[])
            //{
            //    SetParameter((float1[])value);
            //    return;
            //}
            //if (value is float2[])
            //{
            //    SetParameter((float2[])value);
            //    return;
            //}
            //if (value is float3[])
            //{
            //    SetParameter((float3[])value);
            //    return;
            //}
            //if (value is float4[])
            //{
            //    SetParameter((float4[])value);
            //    return;
            //}
            //if (value is double1[])
            //{
            //    SetParameter((double1[])value);
            //    return;
            //}
            //if (value is double2[])
            //{
            //    SetParameter((double2[])value);
            //    return;
            //}
            //if (value is cuDoubleComplex[])
            //{
            //    SetParameter((cuDoubleComplex[])value);
            //    return;
            //}
            //if (value is cuDoubleReal[])
            //{
            //    SetParameter((cuDoubleReal[])value);
            //    return;
            //}
            //if (value is cuFloatComplex[])
            //{
            //    SetParameter((cuFloatComplex[])value);
            //    return;
            //}
            //if (value is cuFloatReal[])
            //{
            //    SetParameter((cuFloatReal[])value);
            //    return;
            }*/
            #endregion

            if (value is CUdeviceptr) 
            {
                SetParameter((CUdeviceptr)value);
                return;
            }
            //if (value is IntPtr) 
            //{
            //    SetParameter((IntPtr)value);
            //    return;
            //}
            throw new CudaException(CUResult.ErrorInvalidValue, "Invalid value type in kernel launch method", null);
        }
        #endregion

        #region Run methods
        /// <summary>
        /// Executes the kernel on the device<para/>
        /// Kernel paramteres must have been set using <see cref="SetParameter(byte)"/> before calling this <see cref="Run()"/> method.<para/>
        /// Or use the the <see cref="Run(object[])"/> method without a call to <see cref="SetParameter(byte)"/> in advance.
        /// </summary>
        /// <returns>Time of execution in milliseconds (using GPU counter)</returns>
        public float Run()
        {
            res = (CUResult)_dispatcher.Invoke(new cuParamSetSize(DriverAPINativeMethods.ParameterManagement.cuParamSetSize), _function, (uint)_paramOffset);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter Size: {3}", DateTime.Now, "cuParamSetSize", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetSize(_function, (uint)_paramOffset);
            if (res != CUResult.Success) throw new CudaException(res);

            res = (CUResult)_dispatcher.Invoke(new cuFuncSetBlockShape(DriverAPINativeMethods.FunctionManagement.cuFuncSetBlockShape), _function, (int)_blockDim.x, (int)_blockDim.y, (int)_blockDim.z);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuFuncSetBlockShape", res));
            //res = DriverAPI.FunctionManagement.cuFuncSetBlockShape(_function, (int)_blockDim.x, (int)_blockDim.y, (int)_blockDim.z);
            if (res != CUResult.Success) throw new CudaException(res);

            res = (CUResult)_dispatcher.Invoke(new cuFuncSetSharedSize(DriverAPINativeMethods.FunctionManagement.cuFuncSetSharedSize), _function, _sharedMemSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuFuncSetSharedSize", res));
            //res = DriverAPI.FunctionManagement.cuFuncSetSharedSize(_function, _sharedMemSize);
            if (res != CUResult.Success) throw new CudaException(res);

            res = (CUResult)_dispatcher.Invoke(new cuCtxSynchronize(DriverAPINativeMethods.ContextManagement.cuCtxSynchronize));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuCtxSynchronize", res));
            //res = DriverAPI.ContextManagement.cuCtxSynchronize();
            if (res != CUResult.Success) throw new CudaException(res);

            //Init Cuda events to measure execution time
            CUevent EventStart = new CUevent();
            CUevent EventEnd = new CUevent();
            CUstream stream = new CUstream();
            object[] paramEventStart = { EventStart, BasicTypes.CUEventFlags.Default };
            object[] paramEventEnd = { EventEnd, BasicTypes.CUEventFlags.Default };

            _dispatcher.Invoke(new cuEventCreate(DriverAPINativeMethods.Events.cuEventCreate), paramEventStart);
            _dispatcher.Invoke(new cuEventCreate(DriverAPINativeMethods.Events.cuEventCreate), paramEventEnd);
            //DriverAPI.Events.cuEventCreate(ref EventStart, (uint)BasicTypes.CUEventFlags.Default);
            //DriverAPI.Events.cuEventCreate(ref EventEnd, (uint)BasicTypes.CUEventFlags.Default);
            EventStart = (CUevent)paramEventStart[0];
            EventEnd = (CUevent)paramEventEnd[0];


            _dispatcher.Invoke(new cuStreamQuery(DriverAPINativeMethods.Streams.cuStreamQuery), stream);
            //DriverAPI.Streams.cuStreamQuery(stream);
            _dispatcher.Invoke(new cuEventRecord(DriverAPINativeMethods.Events.cuEventRecord), EventStart, stream);
            //DriverAPI.Events.cuEventRecord(EventStart, stream);

            //Launch the kernel
            res = (CUResult)_dispatcher.Invoke(new cuLaunchGrid(DriverAPINativeMethods.Launch.cuLaunchGrid), _function, (int)_gridDim.x, (int)_gridDim.y);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuLaunchGrid", res));
            //res = DriverAPI.Launch.cuLaunchGrid(_function, (int)_gridDim.x, (int)_gridDim.y);
            if (res != CUResult.Success) throw new CudaException(res);

            //wait till kernel finished
            res = (CUResult)_dispatcher.Invoke(new cuCtxSynchronize(DriverAPINativeMethods.ContextManagement.cuCtxSynchronize));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuCtxSynchronize", res));
            //res = DriverAPI.ContextManagement.cuCtxSynchronize();
            if (res != CUResult.Success) throw new CudaException(res);

            //reset the parameter stack
            _paramOffset = 0;

            //Get elapsed time
            float ms = 0;
            _dispatcher.Invoke(new cuStreamQuery(DriverAPINativeMethods.Streams.cuStreamQuery), stream);
            //DriverAPI.Streams.cuStreamQuery(stream);
            _dispatcher.Invoke(new cuEventRecord(DriverAPINativeMethods.Events.cuEventRecord), EventEnd, stream);
            //DriverAPI.Events.cuEventRecord(EventEnd, stream);
            _dispatcher.Invoke(new cuEventSynchronize(DriverAPINativeMethods.Events.cuEventSynchronize), EventEnd);
            //DriverAPI.Events.cuEventSynchronize(EventEnd);    
            object[] paramEventElapsedTime = { ms, EventStart, EventEnd };
            _dispatcher.Invoke(new cuEventElapsedTime(DriverAPINativeMethods.Events.cuEventElapsedTime), paramEventElapsedTime);
            //DriverAPI.Events.cuEventElapsedTime(ref ms, EventStart, EventEnd); 
            ms = (float)paramEventElapsedTime[0];
            return ms;
        }

        /// <summary>
        /// Executes the kernel on the device
        /// </summary>
        /// <param name="parameters">Parameters as given by the kernel</param>
        /// <returns>Time of execution in milliseconds (using GPU counter)</returns>
        public float Run(params object[] parameters)
        {
            //reset the parameter stack
            _paramOffset = 0;

            //Set the parameters
            foreach (object parameter in parameters)
            {
                setParameterObject(parameter);
            }

            res = (CUResult)_dispatcher.Invoke(new cuParamSetSize(DriverAPINativeMethods.ParameterManagement.cuParamSetSize), _function, (uint)_paramOffset);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Parameter Size: {3}", DateTime.Now, "cuParamSetSize", res, _paramOffset));   
            //res = DriverAPI.ParameterManagement.cuParamSetSize(_function, (uint)_paramOffset);
            if (res != CUResult.Success) throw new CudaException(res);

            res = (CUResult)_dispatcher.Invoke(new cuFuncSetBlockShape(DriverAPINativeMethods.FunctionManagement.cuFuncSetBlockShape), _function, (int)_blockDim.x, (int)_blockDim.y, (int)_blockDim.z);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuFuncSetBlockShape", res));
            //res = DriverAPI.FunctionManagement.cuFuncSetBlockShape(_function, (int)_blockDim.x, (int)_blockDim.y, (int)_blockDim.z);
            if (res != CUResult.Success) throw new CudaException(res);

            res = (CUResult)_dispatcher.Invoke(new cuFuncSetSharedSize(DriverAPINativeMethods.FunctionManagement.cuFuncSetSharedSize), _function, _sharedMemSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuFuncSetSharedSize", res));
            //res = DriverAPI.FunctionManagement.cuFuncSetSharedSize(_function, _sharedMemSize);
            if (res != CUResult.Success) throw new CudaException(res);

            res = (CUResult)_dispatcher.Invoke(new cuCtxSynchronize(DriverAPINativeMethods.ContextManagement.cuCtxSynchronize));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuCtxSynchronize", res));
            //res = DriverAPI.ContextManagement.cuCtxSynchronize();
            if (res != CUResult.Success) throw new CudaException(res);

            //Init Cuda events to measure execution time
            CUevent EventStart = new CUevent();
            CUevent EventEnd = new CUevent();
            CUstream stream = new CUstream();
            object[] paramEventStart = { EventStart, BasicTypes.CUEventFlags.Default };
            object[] paramEventEnd = { EventEnd, BasicTypes.CUEventFlags.Default };

            _dispatcher.Invoke(new cuEventCreate(DriverAPINativeMethods.Events.cuEventCreate), paramEventStart);
            _dispatcher.Invoke(new cuEventCreate(DriverAPINativeMethods.Events.cuEventCreate), paramEventEnd);
            //DriverAPI.Events.cuEventCreate(ref EventStart, (uint)BasicTypes.CUEventFlags.Default);
            //DriverAPI.Events.cuEventCreate(ref EventEnd, (uint)BasicTypes.CUEventFlags.Default);
            EventStart = (CUevent)paramEventStart[0];
            EventEnd = (CUevent)paramEventEnd[0];


            _dispatcher.Invoke(new cuStreamQuery(DriverAPINativeMethods.Streams.cuStreamQuery), stream);
            //DriverAPI.Streams.cuStreamQuery(stream);
            _dispatcher.Invoke(new cuEventRecord(DriverAPINativeMethods.Events.cuEventRecord), EventStart, stream);
            //DriverAPI.Events.cuEventRecord(EventStart, stream);

            //Launch the kernel
            res = (CUResult)_dispatcher.Invoke(new cuLaunchGrid(DriverAPINativeMethods.Launch.cuLaunchGrid), _function, (int)_gridDim.x, (int)_gridDim.y);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuLaunchGrid", res));
            //res = DriverAPI.Launch.cuLaunchGrid(_function, (int)_gridDim.x, (int)_gridDim.y);
            if (res != CUResult.Success) throw new CudaException(res);

            //wait till kernel finished
            res = (CUResult)_dispatcher.Invoke(new cuCtxSynchronize(DriverAPINativeMethods.ContextManagement.cuCtxSynchronize));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuCtxSynchronize", res));
            //res = DriverAPI.ContextManagement.cuCtxSynchronize();
            if (res != CUResult.Success) throw new CudaException(res);

            //reset the parameter stack
            _paramOffset = 0;

            //Get elapsed time
            float ms = 0;
            _dispatcher.Invoke(new cuStreamQuery(DriverAPINativeMethods.Streams.cuStreamQuery), stream);
            //DriverAPI.Streams.cuStreamQuery(stream);
            _dispatcher.Invoke(new cuEventRecord(DriverAPINativeMethods.Events.cuEventRecord), EventEnd, stream);
            //DriverAPI.Events.cuEventRecord(EventEnd, stream);
            _dispatcher.Invoke(new cuEventSynchronize(DriverAPINativeMethods.Events.cuEventSynchronize), EventEnd);
            //DriverAPI.Events.cuEventSynchronize(EventEnd);    
            object[] paramEventElapsedTime = { ms, EventStart, EventEnd };
            _dispatcher.Invoke(new cuEventElapsedTime(DriverAPINativeMethods.Events.cuEventElapsedTime), paramEventElapsedTime);
            //DriverAPI.Events.cuEventElapsedTime(ref ms, EventStart, EventEnd); 
            ms = (float)paramEventElapsedTime[0];
            return ms;
        }
        #endregion

        #region Properties
        /// <summary>
        /// Get or set the thread block dimensions. Block dimenions must be set before the first kernel launch.
        /// </summary>
        public dim3 BlockDimensions
        {
            get { return _blockDim; }
            set { _blockDim = value; }
        }

        /// <summary>
        /// Get or set the thread grid dimensions. Grid dimenions must be set before the first kernel launch.
        /// z component is set to 1
        /// </summary>
        public dim3 GridDimensions
        {
            get { return _gridDim; }
            set
            {
                _gridDim = value;
                _gridDim.z = 1;
            }
        }

        /// <summary>
        /// Dynamic shared memory size in Bytes. Must be set before the first kernel launch.
        /// </summary>
        public uint DynamicSharedMemory
        {
            get { return _sharedMemSize; }
            set { _sharedMemSize = value; }
        }

        /// <summary>
        /// CUFunction
        /// </summary>
        public CUfunction CUFunction
        {
            get { return _function; }
        }

        /// <summary>
        /// CUModule
        /// </summary>
        public CUmodule CUModule
        {
            get { return _module; }
        }
        #endregion
    }
}
