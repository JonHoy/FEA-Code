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
    /// A variable located in CUDA device memory. The data is aligned following <see cref="CudaPitchedDeviceVariable{T}.Pitch"/>
    /// </summary>
    /// <typeparam name="T">variable base type</typeparam>
    [Obsolete("CUDA 4.0 API calls are now thread safe. Dispatched... wrapper classes are of no use now.")]
    public class DispatchedCudaPitchedDeviceVariable<T> : DispatchedCudaBaseClass, IDisposable where T:struct
    {
        CUdeviceptr _devPtr;
        SizeT _height = 0, _width = 0, _pitch = 0;
        SizeT _typeSize = 0;
        CUResult res;
        bool disposed;

        #region Constructors
        /// <summary>
        /// Creates a new CudaPitchedDeviceVariable and allocates the memory on the device
        /// </summary>
        /// <param name="height">In elements</param>
        /// <param name="width">In elements</param>
        /// <param name="dispatcher"></param>
        public DispatchedCudaPitchedDeviceVariable(SizeT height, SizeT width, Dispatcher dispatcher)
            : base(dispatcher)
        {
            _devPtr = new CUdeviceptr();
            _height = height;
            _width = width;
            _typeSize = (SizeT)Marshal.SizeOf(typeof(T));

            object[] paramMemAllocPitch = { _devPtr, _pitch, _typeSize * width, height, _typeSize };

            res = (CUResult)_dispatcher.Invoke(new cuMemAllocPitch_v2(DriverAPINativeMethods.MemoryManagement.cuMemAllocPitch_v2), paramMemAllocPitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocPitch", res));
            //res = DriverAPINativeMethods.MemoryManagement.cuMemAllocPitch(ref _devPtr, ref _pitch, _typeSize * width, height, _typeSize);
            if (res != CUResult.Success) throw new CudaException(res);
            _devPtr = (CUdeviceptr)paramMemAllocPitch[0];
            _pitch = (uint)paramMemAllocPitch[1];
        }

        /// <summary>
        /// Creates a new CudaPitchedDeviceVariable and allocates the memory on the device
        /// </summary>
        /// <param name="height">In elements</param>
        /// <param name="width">In elements</param>
        /// <param name="pack">Group <c>pack</c> elements as one type. E.g. 4 floats in host code to one float4 in device code.</param>
        /// <param name="dispatcher"></param>
        public DispatchedCudaPitchedDeviceVariable(SizeT height, SizeT width, SizeT pack, Dispatcher dispatcher)
            : base(dispatcher)
        {
            _devPtr = new CUdeviceptr();
            _height = height;
            _width = width;
            _typeSize = (SizeT)Marshal.SizeOf(typeof(T)) * pack;


            object[] paramMemAllocPitch = { _devPtr, _pitch, _typeSize * width, height, _typeSize };

            res = (CUResult)_dispatcher.Invoke(new cuMemAllocPitch_v2(DriverAPINativeMethods.MemoryManagement.cuMemAllocPitch_v2), paramMemAllocPitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocPitch", res));
            //res = DriverAPINativeMethods.MemoryManagement.cuMemAllocPitch(ref _devPtr, ref _pitch, _typeSize * width, height, _typeSize);
            if (res != CUResult.Success) throw new CudaException(res);
            _devPtr = (CUdeviceptr)paramMemAllocPitch[0];
            _pitch = (uint)paramMemAllocPitch[1];
        }

        /// <summary>
        /// Creates a new CudaPitchedDeviceVariable from an existing CUdeviceptr
        /// </summary>
        /// <param name="devPtr">devPtr must be allocated on device using cuMemAllocPitch or must be pitched correctly manually</param>
        /// <param name="height">In elements</param>
        /// <param name="width">In elements</param>
        /// <param name="pitch">In bytes</param>
        /// <param name="dispatcher"></param>
        public DispatchedCudaPitchedDeviceVariable(CUdeviceptr devPtr, uint height, uint width, uint pitch, Dispatcher dispatcher)
            : base(dispatcher)
        {
            _devPtr = devPtr;
            _height = height;
            _width = width;
            _pitch = pitch;
            _typeSize = (uint)Marshal.SizeOf(typeof(T));
        }
        /// <summary>
        /// For dispose
        /// </summary>
        ~DispatchedCudaPitchedDeviceVariable()
        {
            Dispose (false);            
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
        protected virtual void Dispose (bool fDisposing)
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
        /// Copy from Host to device memory
        /// </summary>
        /// <param name="hostSrc">Source</param>
        /// <param name="width">Width in bytes</param>
        /// <param name="height">Height in elements</param>
        public void CopyToDevice(IntPtr hostSrc, SizeT width, SizeT height)
        {
            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.srcHost = hostSrc;
            copyParams.srcMemoryType = CUMemoryType.Host;
            copyParams.dstDevice = _devPtr;
            copyParams.dstMemoryType = CUMemoryType.Device;
            copyParams.dstPitch = _pitch;
            copyParams.Height = height;
            copyParams.WidthInBytes = width;

            object[] paramMemcpy2D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy2D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2), paramMemcpy2D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpy2D(ref copyParams);
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
            copyParams.dstPitch = _pitch;
            copyParams.Height = _height;
            copyParams.WidthInBytes = _width * _typeSize;

            object[] paramMemcpy2D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy2D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2), paramMemcpy2D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpy2D(ref copyParams);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy from device to device memory
        /// </summary>
        /// <param name="deviceSrc">Source</param>
        public void CopyToDevice(DispatchedCudaDeviceVariable<T> deviceSrc)
        {
            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.srcDevice = deviceSrc.DevicePointer;
            copyParams.srcMemoryType = CUMemoryType.Device;
            copyParams.dstDevice = _devPtr;
            copyParams.dstMemoryType = CUMemoryType.Device;
            copyParams.dstPitch = _pitch;
            copyParams.Height = _height;
            copyParams.WidthInBytes = _width * _typeSize;

            object[] paramMemcpy2D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy2D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2), paramMemcpy2D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpy2D(ref copyParams);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy from device to device memory
        /// </summary>
        /// <param name="deviceSrc">Source</param>
        public void CopyToDevice(CUdeviceptr deviceSrc)
        {
            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.srcDevice = deviceSrc;
            copyParams.srcMemoryType = CUMemoryType.Device;
            copyParams.dstDevice = _devPtr;
            copyParams.dstMemoryType = CUMemoryType.Device;
            copyParams.dstPitch = _pitch;
            copyParams.Height = _height;
            copyParams.WidthInBytes = _width * _typeSize;

            object[] paramMemcpy2D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy2D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2), paramMemcpy2D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpy2D(ref copyParams);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="hostDest">IntPtr to destination in host memory</param>
        /// <param name="width">Width in bytes</param>
        /// <param name="height">Height in elements</param>
        public void CopyToHost(IntPtr hostDest, SizeT width, SizeT height)
        {
            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.dstHost = hostDest;
            copyParams.dstMemoryType = CUMemoryType.Host;
            copyParams.srcDevice = _devPtr;
            copyParams.srcMemoryType = CUMemoryType.Device;
            copyParams.srcPitch = _pitch;
            copyParams.Height = height;
            copyParams.WidthInBytes = width;

            object[] paramMemcpy2D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy2D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2), paramMemcpy2D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpy2D(ref copyParams);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy from Host to device memory
        /// </summary>
        /// <param name="hostSrc">Source</param>
        /// <param name="width">Width in elements</param>
        /// <param name="height">Height in elements</param>
        public void CopyToDevice(T[] hostSrc, SizeT width, SizeT height)
        {
            GCHandle handle = GCHandle.Alloc(hostSrc, GCHandleType.Pinned);

            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.srcHost = handle.AddrOfPinnedObject();
            copyParams.srcMemoryType = CUMemoryType.Host;
            copyParams.dstDevice = _devPtr;
            copyParams.dstMemoryType = CUMemoryType.Device;
            copyParams.dstPitch = _pitch;
            copyParams.Height = height;
            copyParams.WidthInBytes = width * (uint)Marshal.SizeOf(typeof(T));

            object[] paramMemcpy2D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy2D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2), paramMemcpy2D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpy2D(ref copyParams);
            handle.Free();
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="hostDest">Destination</param>
        /// <param name="width">Width in elements</param>
        /// <param name="height">Height in elements</param>
        public void CopyToHost(T[] hostDest, SizeT width, SizeT height)
        {
            GCHandle handle = GCHandle.Alloc(hostDest, GCHandleType.Pinned);

            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.dstHost = handle.AddrOfPinnedObject();
            copyParams.dstMemoryType = CUMemoryType.Host;
            copyParams.srcDevice = _devPtr;
            copyParams.srcMemoryType = CUMemoryType.Device;
            copyParams.srcPitch = _pitch;
            copyParams.Height = height;
            copyParams.WidthInBytes = width * (uint)Marshal.SizeOf(typeof(T));

            object[] paramMemcpy2D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy2D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2), paramMemcpy2D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpy2D(ref copyParams);
            handle.Free();
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy from Host to device memory
        /// </summary>
        /// <param name="hostSrc">Source</param>
        public void CopyToDevice(IntPtr hostSrc)
        {
            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.srcHost = hostSrc;
            copyParams.srcMemoryType = CUMemoryType.Host;
            copyParams.dstDevice = _devPtr;
            copyParams.dstMemoryType = CUMemoryType.Device;
            copyParams.dstPitch = _pitch;
            copyParams.Height = _height;
            copyParams.WidthInBytes = _width * _typeSize;

            object[] paramMemcpy2D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy2D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2), paramMemcpy2D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpy2D(ref copyParams);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="hostDest">IntPtr to destination in host memory</param>
        public void CopyToHost(IntPtr hostDest)
        {
            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.dstHost = hostDest;
            copyParams.dstMemoryType = CUMemoryType.Host;
            copyParams.srcDevice = _devPtr;
            copyParams.srcMemoryType = CUMemoryType.Device;
            copyParams.srcPitch = _pitch;
            copyParams.Height = _height;
            copyParams.WidthInBytes = _width * _typeSize;

            object[] paramMemcpy2D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy2D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2), paramMemcpy2D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpy2D(ref copyParams);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy from Host to device memory
        /// </summary>
        /// <param name="hostSrc">Source</param>
        public void CopyToDevice(T[] hostSrc)
        {
            GCHandle handle = GCHandle.Alloc(hostSrc, GCHandleType.Pinned);

            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.srcHost = handle.AddrOfPinnedObject();
            copyParams.srcMemoryType = CUMemoryType.Host;
            copyParams.dstDevice = _devPtr;
            copyParams.dstMemoryType = CUMemoryType.Device;
            copyParams.dstPitch = _pitch;
            copyParams.Height = _height;
            copyParams.WidthInBytes = _width * _typeSize;

            object[] paramMemcpy2D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy2D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2), paramMemcpy2D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpy2D(ref copyParams);
            handle.Free();
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="hostDest">Destination</param>
        public void CopyToHost(T[] hostDest)
        {
            GCHandle handle = GCHandle.Alloc(hostDest, GCHandleType.Pinned);

            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.dstHost = handle.AddrOfPinnedObject();
            copyParams.dstMemoryType = CUMemoryType.Host;
            copyParams.srcDevice = _devPtr;
            copyParams.srcMemoryType = CUMemoryType.Device;
            copyParams.srcPitch = _pitch;
            copyParams.Height = _height;
            copyParams.WidthInBytes = _width * _typeSize;

            object[] paramMemcpy2D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy2D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2), paramMemcpy2D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpy2D(ref copyParams);
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
        /// Width in elements
        /// </summary>
        public SizeT Width
        {
            get { return _width; }
        }

        /// <summary>
        /// Width in bytes
        /// </summary>
        public SizeT WidthInBytes
        {
            get { return _width * _typeSize; }
        }

        /// <summary>
        /// Height in elements
        /// </summary>
        public SizeT Height
        {
            get { return _height; }
        }

        /// <summary>
        /// Pitch in bytes
        /// </summary>
        public SizeT Pitch
        {
            get { return _pitch; }
        }

        /// <summary>
        /// Total size in bytes
        /// </summary>
        public SizeT TotalSizeInBytes
        {
            get { return _pitch * _height; }
        }
        #endregion
    }
}
