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
    /// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.
    /// </summary>
    /// <typeparam name="T">variable base type</typeparam>
    [Obsolete("CUDA 4.0 API calls are now thread safe. Dispatched... wrapper classes are of no use now.")]
    public class CudaPageLockedHostMemory<T> : DispatchedCudaBaseClass, IDisposable where T : struct
    {
        IntPtr _intPtr;
        SizeT _size = 0;
        SizeT _typeSize = 0;
        CUResult res;
        bool disposed;

        #region Constructors
        /// <summary>
        /// Creates a new CudaPageLockedHostMemory and allocates the memory on the device. Using cuMemHostAlloc
        /// </summary>
        /// <param name="size">In elements</param>
        /// <param name="flags"></param>
        /// <param name="dispatcher"></param>
        public CudaPageLockedHostMemory(SizeT size, CUMemHostAllocFlags flags, Dispatcher dispatcher)
            : base(dispatcher)
        {
            _intPtr = new IntPtr();
            _size = size;
            _typeSize = (SizeT)Marshal.SizeOf(typeof(T));

            object[] paramMemHostAlloc = { _intPtr, _typeSize * size, flags };

            res = (CUResult)_dispatcher.Invoke(new cuMemHostAlloc(DriverAPINativeMethods.MemoryManagement.cuMemHostAlloc), paramMemHostAlloc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostAlloc", res));
            //res = DriverAPINativeMethods.MemoryManagement.cuMemHostAlloc(ref _intPtr, _typeSize * size, flags);
            if (res != CUResult.Success) throw new CudaException(res);
            _intPtr = (IntPtr)paramMemHostAlloc[0];
        }

        /// <summary>
        /// Creates a new CudaPageLockedHostMemory and allocates the memory on the device. Using cuMemAllocHost
        /// </summary>
        /// <param name="size">In elements</param>
        /// <param name="dispatcher"></param>
        public CudaPageLockedHostMemory(SizeT size, Dispatcher dispatcher)
            : base(dispatcher)
        {
            _intPtr = new IntPtr();
            _size = size;
            _typeSize = (SizeT)Marshal.SizeOf(typeof(T));

            object[] paramMemAllocHost = { _intPtr, _typeSize * size };

            res = (CUResult)_dispatcher.Invoke(new cuMemAllocHost_v2(DriverAPINativeMethods.MemoryManagement.cuMemAllocHost_v2), paramMemAllocHost);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocHost", res));
            //res = DriverAPINativeMethods.MemoryManagement.cuMemAllocHost(ref _intPtr, _typeSize * size);
            if (res != CUResult.Success) throw new CudaException(res);
            _intPtr = (IntPtr)paramMemAllocHost[0];
        }

        /// <summary>
        /// Creates a new CudaPageLockedHostMemory from an existing IntPtr
        /// </summary>
        /// <param name="devPtr"></param>
        /// <param name="size">In elements</param>
        /// <param name="dispatcher"></param>
        public CudaPageLockedHostMemory(IntPtr devPtr, SizeT size, Dispatcher dispatcher)
            : base(dispatcher)
        {
            _intPtr = devPtr;
            _size = size;
            _typeSize = (SizeT)Marshal.SizeOf(typeof(T));
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~CudaPageLockedHostMemory()
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
                    _dispatcher.Invoke(new cuMemFreeHost(DriverAPINativeMethods.MemoryManagement.cuMemFreeHost), _intPtr);
                disposed = true;
            }
        }
        #endregion

        #region Properties
        /// <summary>
        /// Pointer to pinned host memory.
        /// </summary>
        public IntPtr PinnedHostPointer
        {
            get { return _intPtr; }
        }

        /// <summary>
        /// Size in bytes
        /// </summary>
        public SizeT SizeInBytes
        {
            get { return _size * _typeSize; }
        }

        /// <summary>
        /// Size in elements
        /// </summary>
        public SizeT Size
        {
            get { return _size; }
        }
        #endregion

        #region Synchron Copy Methods
        /// <summary>
        /// Synchron copy host to device
        /// </summary>
        /// <param name="devicePtr"></param>
        public void SynchronCopyToDevice(CUdeviceptr devicePtr)
        {
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDIntPtr(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), devicePtr, this._intPtr, SizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpyHtoD(devicePtr, this._intPtr, SizeInBytes);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Synchron copy host to device
        /// </summary>
        /// <param name="devicePtr"></param>
        public void SynchronCopyToDevice(CudaDeviceVariable<T> devicePtr)
        {
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDIntPtr(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), devicePtr.DevicePointer, this._intPtr, SizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpyHtoD(devicePtr, this._intPtr, SizeInBytes);
            if (res != CUResult.Success) throw new CudaException(res);
        }


        /// <summary>
        /// Synchron copy device to host
        /// </summary>
        /// <param name="devicePtr"></param>
        public void SynchronCopyToHost(CUdeviceptr devicePtr)
        {
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHIntPtr(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), this._intPtr, devicePtr, SizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpyDtoH(this._intPtr, devicePtr, SizeInBytes);
            if (res != CUResult.Success) throw new CudaException(res);
        }


        /// <summary>
        /// Synchron copy device to host
        /// </summary>
        /// <param name="devicePtr"></param>
        public void SynchronCopyToHost(CudaDeviceVariable<T> devicePtr)
        {
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHIntPtr(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), this._intPtr, devicePtr.DevicePointer, SizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpyDtoH(this._intPtr, devicePtr, SizeInBytes);
            if (res != CUResult.Success) throw new CudaException(res);
        }
        #endregion

        #region Asynchron Copy Methods
        /// <summary>
        /// Asynchron Copy host to device
        /// </summary>
        /// <param name="devicePtr"></param>
        /// <param name="stream"></param>
        public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
        {
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDAsync(DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2), devicePtr, _intPtr, SizeInBytes, stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
            //res = DriverAPINativeMethods.AsynchronousMemcpy.cuMemcpyHtoDAsync(devicePtr, _intPtr, SizeInBytes, stream);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Asynchron copy host to 1D Array
        /// </summary>
        /// <param name="deviceArray"></param>
        /// <param name="stream"></param>
        /// <param name="offset"></param>
        public void AsyncCopyToArray1D(CUarray deviceArray, uint offset, CUstream stream)
        {
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAAsync(DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2), deviceArray, offset, this._intPtr, SizeInBytes, stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
            //res = DriverAPINativeMethods.AsynchronousMemcpy.cuMemcpyHtoAAsync(deviceArray, offset, this._intPtr, SizeInBytes, stream);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Asynchron copy host to 1D Array
        /// </summary>
        /// <param name="deviceArray"></param>
        /// <param name="stream"></param>
        public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
        {
            AsyncCopyToArray1D(deviceArray, 0, stream);
        }

        /// <summary>
        /// Asynchron copy host to 2D Array
        /// </summary>
        /// <param name="deviceArray"></param>
        /// <param name="widthInBytes"></param>
        /// <param name="height"></param>
        /// <param name="stream"></param>
        public void AsyncCopyToArray2D(CUarray deviceArray, SizeT widthInBytes, SizeT height, CUstream stream)
        {
            CUDAMemCpy2D cpyProps = new CUDAMemCpy2D();
            cpyProps.dstArray = deviceArray;
            cpyProps.dstMemoryType = CUMemoryType.Array;
            cpyProps.srcHost = _intPtr;
            cpyProps.srcMemoryType = CUMemoryType.Host;
            cpyProps.WidthInBytes = widthInBytes;
            cpyProps.Height = height;

            object[] paramCpy = { cpyProps, stream };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy2DAsync(DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpy2DAsync_v2), paramCpy);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2DAsync", res));
            //res = DriverAPINativeMethods.AsynchronousMemcpy.cuMemcpy2DAsync(ref cpyProps, stream);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Asynchron copy host to 3D array
        /// </summary>
        /// <param name="deviceArray"></param>
        /// <param name="widthInBytes"></param>
        /// <param name="height"></param>
        /// <param name="depth"></param>
        /// <param name="stream"></param>
        public void AsyncCopyToArray3D(CUarray deviceArray, SizeT widthInBytes, SizeT height, SizeT depth, CUstream stream)
        {
            CUDAMemCpy3D cpyProps = new CUDAMemCpy3D();
            cpyProps.dstArray = deviceArray;
            cpyProps.dstMemoryType = CUMemoryType.Array;
            cpyProps.srcHost = _intPtr;
            cpyProps.srcMemoryType = CUMemoryType.Host;
            cpyProps.WidthInBytes = widthInBytes;
            cpyProps.Height = height;
            cpyProps.Depth = depth;

            object[] paramCpy = { cpyProps, stream };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy3DAsync(DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpy3DAsync_v2), paramCpy);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy3DAsync", res));
            //res = DriverAPINativeMethods.AsynchronousMemcpy.cuMemcpy3DAsync(ref cpyProps, stream);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Asynchron copy host to 1D Array
        /// </summary>
        /// <param name="array"></param>
        /// <param name="stream"></param>
        /// <param name="offset"></param>
        public void AsyncCopyToArray1D(CudaArray1D array, SizeT offset, CUstream stream)
        {
            AsyncCopyToArray1D(array.CUArray, offset, stream);
        }

        /// <summary>
        /// Asynchron copy host to 1D Array
        /// </summary>
        /// <param name="array"></param>
        /// <param name="stream"></param>
        public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
        {
            AsyncCopyToArray1D(array.CUArray, 0, stream);
        }

        /// <summary>
        /// Asynchron copy host to 2D Array
        /// </summary>
        /// <param name="array"></param>
        /// <param name="stream"></param>
        public void AsyncCopyToArray2D(CudaArray2D array, CUstream stream)
        {
            AsyncCopyToArray2D(array.CUArray, array.WidthInBytes, array.Height, stream);
        }

        /// <summary>
        /// Asynchron copy host to 3D Array
        /// </summary>
        /// <param name="array"></param>
        /// <param name="stream"></param>
        public void AsyncCopyToArray3D(CudaArray3D array, CUstream stream)
        {
            AsyncCopyToArray3D(array.CUArray, array.WidthInBytes, array.Height, array.Depth, stream);
        }



        /// <summary>
        /// Asynchron copy device to host
        /// </summary>
        /// <param name="devicePtr"></param>
        /// <param name="stream"></param>
        public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
        {
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHAsync(DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2), _intPtr, devicePtr, SizeInBytes, stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
            //res = DriverAPINativeMethods.AsynchronousMemcpy.cuMemcpyDtoHAsync(_intPtr, devicePtr, SizeInBytes, stream);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Asynchron copy 1D Array to host
        /// </summary>
        /// <param name="deviceArray"></param>
        /// <param name="stream"></param>
        /// <param name="offset"></param>
        public void AsyncCopyFromArray1D(CUarray deviceArray, SizeT offset, CUstream stream)
        {
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHAsync(DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2), this._intPtr, deviceArray, offset, SizeInBytes, stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
            //res = DriverAPINativeMethods.AsynchronousMemcpy.cuMemcpyAtoHAsync(this._intPtr, deviceArray, offset, SizeInBytes, stream);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Asynchron copy 1D Array to host
        /// </summary>
        /// <param name="deviceArray"></param>
        /// <param name="stream"></param>
        public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
        {
            AsyncCopyFromArray1D(deviceArray, 0, stream);
        }

        /// <summary>
        /// Asynchron copy 2D Array to host
        /// </summary>
        /// <param name="deviceArray"></param>
        /// <param name="widthInBytes"></param>
        /// <param name="height"></param>
        /// <param name="stream"></param>
        public void AsyncCopyFromArray2D(CUarray deviceArray, SizeT widthInBytes, SizeT height, CUstream stream)
        {
            CUDAMemCpy2D cpyProps = new CUDAMemCpy2D();
            cpyProps.srcArray = deviceArray;
            cpyProps.srcMemoryType = CUMemoryType.Array;
            cpyProps.dstHost = _intPtr;
            cpyProps.dstMemoryType = CUMemoryType.Host;
            cpyProps.WidthInBytes = widthInBytes;
            cpyProps.Height = height;

            object[] paramCpy = { cpyProps, stream };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy2DAsync(DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpy2DAsync_v2), paramCpy);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2DAsync", res));
            //res = DriverAPINativeMethods.AsynchronousMemcpy.cuMemcpy2DAsync(ref cpyProps, stream);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Asynchron copy 3D Array to host
        /// </summary>
        /// <param name="deviceArray"></param>
        /// <param name="widthInBytes"></param>
        /// <param name="height"></param>
        /// <param name="depth"></param>
        /// <param name="stream"></param>
        public void AsyncCopyFromArray3D(CUarray deviceArray, SizeT widthInBytes, SizeT height, SizeT depth, CUstream stream)
        {
            CUDAMemCpy3D cpyProps = new CUDAMemCpy3D();
            cpyProps.srcArray = deviceArray;
            cpyProps.srcMemoryType = CUMemoryType.Array;
            cpyProps.dstHost = _intPtr;
            cpyProps.dstMemoryType = CUMemoryType.Host;
            cpyProps.WidthInBytes = widthInBytes;
            cpyProps.Height = height;
            cpyProps.Depth = depth;

            object[] paramCpy = { cpyProps, stream };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy3DAsync(DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpy3DAsync_v2), paramCpy);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy3DAsync", res));
            //res = DriverAPINativeMethods.AsynchronousMemcpy.cuMemcpy3DAsync(ref cpyProps, stream);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Asynchron copy 1D Array to host
        /// </summary>
        /// <param name="array"></param>
        /// <param name="stream"></param>
        /// <param name="offset"></param>
        public void AsyncCopyFromArray1D(CudaArray1D array, SizeT offset, CUstream stream)
        {
            AsyncCopyFromArray1D(array.CUArray, offset, stream);
        }

        /// <summary>
        /// Asynchron copy 1D Array to host
        /// </summary>
        /// <param name="array"></param>
        /// <param name="stream"></param>
        public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
        {
            AsyncCopyFromArray1D(array.CUArray, 0, stream);
        }

        /// <summary>
        /// Asynchron copy 2D Array to host
        /// </summary>
        /// <param name="array"></param>
        /// <param name="stream"></param>
        public void AsyncCopyFromArray2D(CudaArray2D array, CUstream stream)
        {
            AsyncCopyFromArray2D(array.CUArray, array.WidthInBytes, array.Height, stream);
        }

        /// <summary>
        /// Asynchron copy 3D Array to host
        /// </summary>
        /// <param name="array"></param>
        /// <param name="stream"></param>
        public void AsyncCopyFromArray3D(CudaArray3D array, CUstream stream)
        {
            AsyncCopyFromArray3D(array.CUArray, array.WidthInBytes, array.Height, array.Depth, stream);
        }
        #endregion
    }
}
