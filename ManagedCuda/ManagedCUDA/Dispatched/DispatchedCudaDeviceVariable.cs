using System;
using System.Collections.Generic;
using System.Text;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System.Runtime.InteropServices;
using System.Windows.Threading;
using System.Diagnostics;

namespace ManagedCuda.Dispatched
{
    /// <summary>
    /// A variable located in CUDA device memory
    /// </summary>
    /// <typeparam name="T">variable base type</typeparam>
    [Obsolete("CUDA 4.0 API calls are now thread safe. Dispatched... wrapper classes are of no use now.")]
    public class DispatchedCudaDeviceVariable<T> : DispatchedCudaBaseClass, IDisposable where T : struct
    {
        CUdeviceptr _devPtr;
        SizeT _size = 0;
        SizeT _typeSize = 0;
        CUResult res;
        bool disposed;

        #region Constructors
        /// <summary>
        /// Creates a new CudaDeviceVariable and allocates the memory on the device
        /// </summary>
        /// <param name="size">In elements</param>
        /// <param name="dispatcher"></param>
        public DispatchedCudaDeviceVariable(SizeT size, Dispatcher dispatcher)
            : base(dispatcher)
        {
            _devPtr = new CUdeviceptr();
            _size = size;
            _typeSize = (uint)Marshal.SizeOf(typeof(T));

            object[] paramMemAlloc = { _devPtr, _typeSize * size };

            res = (CUResult)_dispatcher.Invoke(new cuMemAlloc_v2(DriverAPINativeMethods.MemoryManagement.cuMemAlloc_v2), paramMemAlloc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAlloc", res));
            //res = DriverAPINativeMethods.MemoryManagement.cuMemAlloc(ref _devPtr, _typeSize * size);
            if (res != CUResult.Success) throw new CudaException(res);
            _devPtr = (CUdeviceptr)paramMemAlloc[0];
        }

        /// <summary>
        /// Creates a new CudaDeviceVariable from an existing CUdeviceptr. The allocated size is gethered via the CUDA API
        /// </summary>
        /// <param name="devPtr"></param>
        /// <param name="dispatcher"></param>
        public DispatchedCudaDeviceVariable(CUdeviceptr devPtr, Dispatcher dispatcher)  
            : base(dispatcher)
        {
            _devPtr = devPtr;
            CUdeviceptr NULL = new CUdeviceptr();

            object[] paramMemGetAddressRange = { NULL, _size, devPtr };

            res = (CUResult)_dispatcher.Invoke(new cuMemGetAddressRange_v2(DriverAPINativeMethods.MemoryManagement.cuMemGetAddressRange_v2), paramMemGetAddressRange);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemGetAddressRange", res));
            //res = DriverAPINativeMethods.MemoryManagement.cuMemGetAddressRange(ref NULL, ref _size, devPtr);
            if (res != CUResult.Success) throw new CudaException(res);
            _typeSize = (SizeT)Marshal.SizeOf(typeof(T));
            _size = (SizeT)paramMemGetAddressRange[1];
            _size = _size / _typeSize;
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~DispatchedCudaDeviceVariable()
        {
            Dispose(false);
        }
        #endregion

        #region Dispose
        /// <summary>
        /// Dispose
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// For IDisposable
        /// </summary>
        /// <param name="fDisposing"></param>
        protected virtual void Dispose(bool fDisposing)
        {
            if (fDisposing && !disposed)
            {
                if (_dispatcher != null)
                    _dispatcher.Invoke(new cuMemFree_v2(DriverAPINativeMethods.MemoryManagement.cuMemFree_v2), _devPtr);
                disposed = true;
            }
        }
        #endregion

        #region Methods
        /// <summary>
        /// Copy data from device to device memory
        /// </summary>
        /// <param name="source">Source pointer to host memory</param>
        public void CopyToDevice(CUdeviceptr source)
        {
            SizeT aSizeInBytes = _size * _typeSize;
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoD(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoD_v2), _devPtr, source, aSizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoD", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpyHtoD(_devPtr, ptr, aSizeInBytes);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to device memory
        /// </summary>
        /// <param name="source">Source pointer to host memory</param>
        public void CopyToDevice(DispatchedCudaDeviceVariable<T> source)
        {
            SizeT aSizeInBytes = _size * _typeSize;
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoD(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoD_v2), _devPtr, source.DevicePointer, aSizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoD", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpyHtoD(_devPtr, ptr, aSizeInBytes);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy from device to device memory
        /// </summary>
        /// <param name="deviceSrc">Source</param>
        public void CopyToDevice(DispatchedCudaPitchedDeviceVariable<T> deviceSrc)
        {
            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.srcDevice = deviceSrc.DevicePointer;
            copyParams.srcMemoryType = CUMemoryType.Device;
            copyParams.srcPitch = deviceSrc.Pitch;
            copyParams.dstDevice = _devPtr;
            copyParams.dstMemoryType = CUMemoryType.Device;
            copyParams.Height = deviceSrc.Height;
            copyParams.WidthInBytes = deviceSrc.WidthInBytes;

            object[] paramMemcpy2D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy2D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2), paramMemcpy2D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpy2D(ref copyParams);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="source">Source pointer to host memory</param>
        public void CopyToDevice(T[] source)
        {
            SizeT aSizeInBytes = (source.LongLength * _typeSize);
            GCHandle handle = GCHandle.Alloc(source, GCHandleType.Pinned);
            IntPtr ptr = handle.AddrOfPinnedObject();
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDIntPtr(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), _devPtr, ptr, aSizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpyHtoD(_devPtr, ptr, aSizeInBytes);
            handle.Free();
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="source">Source pointer to host memory</param>
        public void CopyToDevice(T source)
        {
            SizeT aSizeInBytes = _typeSize;
            GCHandle handle = GCHandle.Alloc(source, GCHandleType.Pinned);
            IntPtr ptr = handle.AddrOfPinnedObject();
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDIntPtr(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), _devPtr, ptr, aSizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpyHtoD(_devPtr, ptr, aSizeInBytes);
            handle.Free();
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination pointer to host memory</param>
        public void CopyToHost(T[] dest)
        {
            SizeT aSizeInBytes = (dest.LongLength * _typeSize);
            GCHandle handle = GCHandle.Alloc(dest, GCHandleType.Pinned);
            IntPtr ptr = handle.AddrOfPinnedObject();
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHIntPtr(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), ptr, _devPtr, aSizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpyDtoH(ptr, _devPtr, aSizeInBytes);
            handle.Free();
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination data in host memory</param>
        public void CopyToHost(T dest)
        {
            SizeT aSizeInBytes = _typeSize;
            GCHandle handle = GCHandle.Alloc(dest, GCHandleType.Pinned);
            IntPtr ptr = handle.AddrOfPinnedObject();
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHIntPtr(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), ptr, _devPtr, aSizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpyDtoH(ptr, _devPtr, aSizeInBytes);
            handle.Free();
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        #endregion

        #region Properties
        /// <summary>
        /// Device pointer
        /// </summary>
        public CUdeviceptr DevicePointer
        {
            get { return _devPtr; }
        }

        /// <summary>
        /// Size in bytes
        /// </summary>
        public SizeT SizeInBytes
        {
            get { return _size * _typeSize; }
        }

        /// <summary>
        /// Type size in bytes
        /// </summary>
        public SizeT TypeSize
        {
            get { return _typeSize; }
        }

        /// <summary>
        /// Size in elements
        /// </summary>
        public SizeT Size
        {
            get { return _size; }
        }
        #endregion
    }
}
