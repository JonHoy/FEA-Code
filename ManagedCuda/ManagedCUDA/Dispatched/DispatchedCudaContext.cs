using System;
using System.Collections.Generic;
using System.Text;
using ManagedCuda.BasicTypes;
using System.Runtime.InteropServices;
using System.Windows.Threading;
using System.Threading;
using System.Diagnostics;

namespace ManagedCuda.Dispatched
{
    /// <summary>
    /// An abstraction layer for the CUDA driver API. As all dispatched managedCUDA-object, the abstraction layer
    /// runs on its own thread to avoid thread <![CDATA[<->]]> CUDA context complications.
    /// </summary>
    [Obsolete("CUDA 4.0 API calls are now thread safe. Dispatched... wrapper classes are of no use now.")]
    public class DispatchedCudaContext : DispatchedCudaBaseClass, IDisposable
    {       
        private CUcontext _context;
        private CUdevice _device;
        private int _deviceID;
        //private Dispatcher _dispatcher;
        private Thread _thread;
        private bool disposed;

        private void runThread()
        {
            Debug.WriteLine(String.Format("Thread " + Thread.CurrentThread.Name + " started..."));
            //This call creates a new dispatcher for the new cudaThread
            _dispatcher = Dispatcher.CurrentDispatcher;
            //Start the event loop for the new cudaThread
            Dispatcher.Run();
        }

        #region Constructors
        /// <summary>
        /// Create a new instace of managed and dispatched Cuda.
        /// Using device with ID 0 and <see cref="CUCtxFlags.BlockingSync"/>
        /// </summary>
        public DispatchedCudaContext() 
            : this(0, CUCtxFlags.BlockingSync)
        {
        
        }

        /// <summary>
        /// Create a new instace of managed and dispatched Cuda.
        /// Using <see cref="CUCtxFlags.BlockingSync"/>
        /// </summary>
        /// <param name="deviceId">DeviceID</param>
        public DispatchedCudaContext(int deviceId)
            : this(deviceId, CUCtxFlags.BlockingSync)
        { 
            
        }

        /// <summary>
        /// Create a new instace of managed and dispatched Cuda
        /// </summary>
        /// <param name="deviceId">DeviceID</param>
        /// <param name="flags">Context creation flags</param>
        public DispatchedCudaContext(int deviceId, CUCtxFlags flags)
        {
            _thread = new Thread(runThread)
            {
                IsBackground = true,
                Name = "Thread for Cuda device " + deviceId.ToString(System.Globalization.CultureInfo.InvariantCulture)
            };
            _thread.SetApartmentState(ApartmentState.STA);
            _thread.Start();

            int timeOut = 0;
            while (_dispatcher == null)
            {
                Thread.Sleep(10);
                timeOut += 1;
                if (timeOut > 100)
                    throw new TimeoutException("Initialization of a new CUDA thread took more than one second. Canceling...");
            }

            _dispatcher = Dispatcher.FromThread(_thread);

            CUResult res;
            int deviceCount = 0;
            object[] paramDeviceCount = { deviceCount };

            res = (CUResult)_dispatcher.Invoke(new cuDeviceGetCount(DriverAPINativeMethods.DeviceManagement.cuDeviceGetCount), paramDeviceCount);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDeviceGetCount", res));
            deviceCount = (int)paramDeviceCount[0];

            if (res == CUResult.ErrorNotInitialized)
            {
                res = (CUResult)_dispatcher.Invoke(new Func<CUInitializationFlags, CUResult>(DriverAPINativeMethods.cuInit), CUInitializationFlags.None);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuInit", res));
                if (res != CUResult.Success)
                    throw new CudaException(res);


                res = (CUResult)_dispatcher.Invoke(new cuDeviceGetCount(DriverAPINativeMethods.DeviceManagement.cuDeviceGetCount), paramDeviceCount);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDeviceGetCount", res));
                deviceCount = (int)paramDeviceCount[0];
                if (res != CUResult.Success)
                    throw new CudaException(res);
            }
            else if (res != CUResult.Success)
                throw new CudaException(res);

            if (deviceCount == 0)
            {
                throw new CudaException(CUResult.ErrorNoDevice, "Cuda initialization error: There is no device supporting CUDA", null);
            }

            _deviceID = deviceId;

            object[] paramDeviceGet = { _device, deviceId };
            res = (CUResult)_dispatcher.Invoke(new cuDeviceGet(DriverAPINativeMethods.DeviceManagement.cuDeviceGet), paramDeviceGet);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDeviceGet", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
            _device = (CUdevice)paramDeviceGet[0];

            object[] paramCtxCreate = { _context, flags, _device };
            res = (CUResult)_dispatcher.Invoke(new cuCtxCreate_v2(DriverAPINativeMethods.ContextManagement.cuCtxCreate_v2), paramCtxCreate);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuCtxCreate", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
            _context = (CUcontext)paramCtxCreate[0];
        }

        /// <summary>
        /// Create a new instance of managed Cuda for a given Direct3D-device.
        /// </summary>
        /// <param name="adapterName">DirectX adapter name</param>
        /// <param name="deviceId">CUDA device 0</param>
        /// <param name="d3DDevice">Direct3D device</param>
        public DispatchedCudaContext(string adapterName, int deviceId, IntPtr d3DDevice)
        {
            _thread = new Thread(runThread)
            {
                IsBackground = true,
                Name = "CudaThread for device " + deviceId.ToString(System.Globalization.CultureInfo.InvariantCulture)
            };
            _thread.SetApartmentState(ApartmentState.STA);
            _thread.Start();

            int timeOut = 0;
            while (_dispatcher == null)
            {
                Thread.Sleep(10);
                timeOut += 1;
                if (timeOut > 100)
                    throw new TimeoutException("Initialization of a new CUDA thread took more than one second! Canceling...");
            }

            _dispatcher = Dispatcher.FromThread(_thread);
         
            CUResult res;
            int deviceCount = 0;

            object[] paramDeviceCount = { deviceCount };

            res = (CUResult)_dispatcher.Invoke(new cuDeviceGetCount(DriverAPINativeMethods.DeviceManagement.cuDeviceGetCount), paramDeviceCount);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDeviceGetCount", res));
            deviceCount = (int)paramDeviceCount[0];

            if (res == CUResult.ErrorNotInitialized)
            {
                res = (CUResult)_dispatcher.Invoke(new Func<CUInitializationFlags, CUResult>(DriverAPINativeMethods.cuInit), CUInitializationFlags.None);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuInit", res));
                if (res != CUResult.Success)
                    throw new CudaException(res);


                res = (CUResult)_dispatcher.Invoke(new cuDeviceGetCount(DriverAPINativeMethods.DeviceManagement.cuDeviceGetCount), paramDeviceCount);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDeviceGetCount", res));
                deviceCount = (int)paramDeviceCount[0];
                if (res != CUResult.Success)
                    throw new CudaException(res);
            }
            else if (res != CUResult.Success)
                throw new CudaException(res);

            if (deviceCount == 0)
            {
                throw new CudaException(CUResult.ErrorNoDevice, "Cuda initialization error: There is no device supporting CUDA", null);
            }

            _deviceID = deviceId;

            object[] paramD3D9GetDevice = {_device, adapterName};
            res = (CUResult)_dispatcher.Invoke(new cuD3D9GetDevice(DirectX9NativeMethods.CUDA3.cuD3D9GetDevice), paramD3D9GetDevice);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuD3D9GetDevice", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
            _device = (CUdevice)paramD3D9GetDevice[0];

            object[] paramD3D9CtxCreate = { _context, _device, CUCtxFlags.SchedAuto, d3DDevice };
            res = (CUResult)_dispatcher.Invoke(new cuD3D9CtxCreate(DirectX9NativeMethods.CUDA3.cuD3D9CtxCreate), paramD3D9CtxCreate);//(ref _context, ref _device, CUCtxFlags.SchedAuto, d3DDevice);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuD3D9CtxCreate", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
            _context = (CUcontext)paramD3D9CtxCreate[0];
            _device = (CUdevice)paramD3D9CtxCreate[1];
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~DispatchedCudaContext()
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
                //Check if _dispatcher is still valid
                if (_dispatcher != null)
                    _dispatcher.Invoke(new cuCtxDetach(DriverAPINativeMethods.ContextManagement.cuCtxDetach), _context);
                disposed = true;
                //res = DriverAPI.ContextManagement.cuCtxDetach(_context);
            }
        }
        #endregion

        #region Methods
        /// <summary>
        /// Gets the context's API version
        /// </summary>
        /// <returns>Version</returns>
        public Version GetAPIVersionOfCurrentContext()
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());
            uint version = 0;
            CUResult res;
            object[] paramCtxGetApiVersion = { _context, version };
            res = (CUResult)_dispatcher.Invoke(new cuCtxGetApiVersion(DriverAPINativeMethods.ContextManagement.cuCtxGetApiVersion), paramCtxGetApiVersion);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuCtxGetApiVersion", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
            return new Version((int)version / 1000, (int)version % 100);
        }

        /// <summary>
        /// Push the CUDA context. Watch out, the CUDA context is assigned to it's own CPU thread.
        /// </summary>
        public void PushContext()
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuCtxPushCurrent(DriverAPINativeMethods.ContextManagement.cuCtxPushCurrent), _context);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuCtxPushCurrent", res));
            //res = DriverAPI.ContextManagement.cuCtxPushCurrent(_context);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Pop the CUDA context. Watch out, the CUDA context is assigned to it's own CPU thread.
        /// </summary>
        public void PopContext()
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuCtxPopCurrent(DriverAPINativeMethods.ContextManagement.cuCtxPopCurrent), _context);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuCtxPopCurrent", res));
            //res = DriverAPI.ContextManagement.cuCtxPopCurrent(ref _context);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Load a CUBIN-module from file
        /// </summary>
        /// <param name="modulePath"></param>
        /// <returns></returns>
        public CUmodule LoadModule(string modulePath)
        {
            CUResult res;
            CUmodule module = new CUmodule();
            object[] paramModuleLoad = { module, modulePath };
            res = (CUResult)_dispatcher.Invoke(new cuModuleLoad(DriverAPINativeMethods.ModuleManagement.cuModuleLoad), paramModuleLoad);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleLoad", res));
            //res = DriverAPI.ModuleManagement.cuModuleLoad(ref module, modulePath);
            if (res != CUResult.Success)
                throw new CudaException(res);
            module = (CUmodule)paramModuleLoad[0];
            return module;
        }

        /// <summary>
        /// Load a PTX module from file
        /// </summary>
        /// <param name="modulePath"></param>
        /// <param name="options"></param>
        /// <param name="values"></param>
        /// <returns></returns>
        public CUmodule LoadModulePTX(string modulePath, CUJITOption[] options, object[] values)
        {
            byte[] image;
            System.IO.FileInfo fi = new System.IO.FileInfo(modulePath);
            if (!fi.Exists) throw new System.IO.FileNotFoundException("", modulePath);

            using (System.IO.BinaryReader br = new System.IO.BinaryReader(fi.OpenRead()))
            {
                image = br.ReadBytes((int)br.BaseStream.Length);
            }
            return LoadModulePTX(image, options, values);
        }

        /// <summary>
        /// Load a ptx module from image as byte[]
        /// </summary>
        /// <param name="moduleImage"></param>
        /// <param name="options"></param>
        /// <param name="values"></param>
        /// <returns></returns>
        public CUmodule LoadModulePTX(byte[] moduleImage, CUJITOption[] options, object[] values)
        {
            CUResult res;
            CUmodule module = new CUmodule();
            IntPtr[] valuesIntPtr = new IntPtr[values.Length];
            GCHandle[] gcHandles = new GCHandle[values.Length];

            #region Convert OptionValues
            for (int i = 0; i < values.Length; i++)
            {                
                switch (options[i])
                { 
                    case CUJITOption.ErrorLogBuffer:
                        gcHandles[i] = GCHandle.Alloc(values[i], GCHandleType.Pinned);
                        valuesIntPtr[i] = gcHandles[i].AddrOfPinnedObject();
                        break;
                    case CUJITOption.InfoLogBuffer:
                        gcHandles[i] = GCHandle.Alloc(values[i], GCHandleType.Pinned);
                        valuesIntPtr[i] = gcHandles[i].AddrOfPinnedObject();
                        break;
                    case CUJITOption.ErrorLogBufferSizeBytes:
                        valuesIntPtr[i] = (IntPtr)(Convert.ToUInt32(values[i], System.Globalization.CultureInfo.InvariantCulture));
                        break;
                    case CUJITOption.InfoLogBufferSizeBytes:
                        valuesIntPtr[i] = (IntPtr)(Convert.ToUInt32(values[i], System.Globalization.CultureInfo.InvariantCulture));
                        break;
                    case CUJITOption.FallbackStrategy:
                        valuesIntPtr[i] = (IntPtr)(Convert.ToUInt32(values[i], System.Globalization.CultureInfo.InvariantCulture));
                        break;
                    case CUJITOption.MaxRegisters:
                        valuesIntPtr[i] = (IntPtr)(Convert.ToUInt32(values[i], System.Globalization.CultureInfo.InvariantCulture));
                        break;
                    case CUJITOption.OptimizationLevel:
                        valuesIntPtr[i] = (IntPtr)(Convert.ToUInt32(values[i], System.Globalization.CultureInfo.InvariantCulture));
                        break;
                    case CUJITOption.Target:
                        valuesIntPtr[i] = (IntPtr)(Convert.ToUInt32(values[i], System.Globalization.CultureInfo.InvariantCulture));
                        break;
                    case CUJITOption.TargetFromContext:
                        valuesIntPtr[i] = (IntPtr)(Convert.ToUInt32(values[i], System.Globalization.CultureInfo.InvariantCulture));
                        break;
                    case CUJITOption.ThreadsPerBlock:
                        valuesIntPtr[i] = (IntPtr)(Convert.ToUInt32(values[i], System.Globalization.CultureInfo.InvariantCulture));
                        break;
                    case CUJITOption.WallTime:
                        valuesIntPtr[i] = new IntPtr();
                        break;

                }
            }
            #endregion

            object[] paramModuleLoadDataEx = { module, moduleImage, (uint)options.Length, options, valuesIntPtr };
            res = (CUResult)_dispatcher.Invoke(new cuModuleLoadDataEx(DriverAPINativeMethods.ModuleManagement.cuModuleLoadDataEx), paramModuleLoadDataEx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleLoadDataEx", res));
            //res = DriverAPI.ModuleManagement.cuModuleLoadDataEx(ref module, moduleImage, (uint)options.Length, options, valuesIntPtr);
            module = (CUmodule)paramModuleLoadDataEx[0];
            valuesIntPtr = (IntPtr[])paramModuleLoadDataEx[4];

            #region Convert back OptionValues
            for (int i = 0; i < values.Length; i++)
            {

                switch (options[i])
                {
                    case CUJITOption.ErrorLogBuffer:
                        gcHandles[i].Free();
                        break;
                    case CUJITOption.InfoLogBuffer:
                        gcHandles[i].Free();
                        break;
                    case CUJITOption.ErrorLogBufferSizeBytes:
                        values[i] = (uint)valuesIntPtr[i];
                        break;
                    case CUJITOption.InfoLogBufferSizeBytes:
                        values[i] = (uint)valuesIntPtr[i];
                        break;
                    case CUJITOption.ThreadsPerBlock:
                        values[i] = (uint)valuesIntPtr[i];
                        break;
                    case CUJITOption.WallTime:
                        uint test = (uint)valuesIntPtr[i];
                        byte[] bytes = BitConverter.GetBytes(test);
                        values[i] = BitConverter.ToSingle(bytes, 0);
                        break;

                }
            }
            #endregion

            if (res != CUResult.Success)
                throw new CudaException(res);
            
            return module;
        }

        /// <summary>
        /// unload module
        /// </summary>
        /// <param name="module"></param>
        public void UnloadModule(CUmodule module)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuModuleUnload(DriverAPINativeMethods.ModuleManagement.cuModuleUnload), module);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuModuleUnload", res));
            //res = DriverAPI.ModuleManagement.cuModuleUnload(aModule);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Allocate memory on device
        /// </summary>
        /// <param name="sizeInBytes"></param>
        /// <returns></returns>
        public CUdeviceptr AllocateMemory(SizeT sizeInBytes)
        {
            CUResult res;
            CUdeviceptr dBuffer = new CUdeviceptr();

            object[] paramMemAlloc = { dBuffer, sizeInBytes };
            res = (CUResult)_dispatcher.Invoke(new cuMemAlloc_v2(DriverAPINativeMethods.MemoryManagement.cuMemAlloc_v2), paramMemAlloc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAlloc", res));
            //res = DriverAPI.MemoryManagement.cuMemAlloc(ref dBuffer, sizeInBytes);
            if (res != CUResult.Success)
                throw new CudaException(res);
            dBuffer = (CUdeviceptr)paramMemAlloc[0];
            return dBuffer;
        }

        #region Clear memory
        /// <summary>
        /// SetMemory (cuMemsetD8)
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="value"></param>
        /// <param name="sizeInBytes"></param>
        public void ClearMemory(CUdeviceptr ptr, byte value, SizeT sizeInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemsetD8_v2(DriverAPINativeMethods.Memset.cuMemsetD8_v2), ptr, value, sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemsetD8", res));
            //res = DriverAPI.Memset.cuMemsetD8(aPtr, aValue, N);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// SetMemory (cuMemsetD16)
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="value"></param>
        /// <param name="sizeInBytes"></param>
        public void ClearMemory(CUdeviceptr ptr, ushort value, SizeT sizeInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemsetD16_v2(DriverAPINativeMethods.Memset.cuMemsetD16_v2), ptr, value, sizeInBytes / sizeof(ushort));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemsetD16", res));
            //res = DriverAPI.Memset.cuMemsetD16(aPtr, aValue, sizeInBytes / sizeof(ushort));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// SetMemory (cuMemsetD32)
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="value"></param>
        /// <param name="sizeInBytes"></param>
        public void ClearMemory(CUdeviceptr ptr, uint value, SizeT sizeInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemsetD32_v2(DriverAPINativeMethods.Memset.cuMemsetD32_v2), ptr, value, sizeInBytes / sizeof(uint));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemsetD32", res));
            //res = DriverAPI.Memset.cuMemsetD32(aPtr, aValue, sizeInBytes / sizeof(uint));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// SetMemory (cuMemset2DD8)
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="value"></param>
        /// <param name="pitch"></param>
        /// <param name="height"></param>
        /// <param name="width"></param>
        public void ClearMemory(CUdeviceptr ptr, byte value, SizeT pitch, SizeT width, SizeT height)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemsetD2D8_v2(DriverAPINativeMethods.Memset.cuMemsetD2D8_v2), ptr, pitch, value, width, height);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemsetD2D8", res));
            //res = DriverAPI.Memset.cuMemsetD2D8(aPtr, aPitch, aValue, aWidth, aHeight);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// SetMemory (cuMemset2DD16)
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="value"></param>
        /// <param name="pitch"></param>
        /// <param name="height"></param>
        /// <param name="width"></param>
        public void ClearMemory(CUdeviceptr ptr, ushort value, SizeT pitch, SizeT width, SizeT height)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemsetD2D16_v2(DriverAPINativeMethods.Memset.cuMemsetD2D16_v2), ptr, pitch, value, width, height);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemsetD2D16", res));
            //res = DriverAPI.Memset.cuMemsetD2D16(aPtr, aValue, aValue, aWidth, aHeight);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// SetMemory (cuMemset2DD32)
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="value"></param>
        /// <param name="pitch"></param>
        /// <param name="height"></param>
        /// <param name="width"></param>
        public void ClearMemory(CUdeviceptr ptr, uint value, SizeT pitch, SizeT width, SizeT height)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemsetD2D32_v2(DriverAPINativeMethods.Memset.cuMemsetD2D32_v2), ptr, pitch, value, width, height);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemsetD2D32", res));
            //res = DriverAPI.Memset.cuMemsetD2D32(aPtr, aValue, aValue, aWidth, aHeight);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        #endregion

        /// <summary>
        /// Free device memory
        /// </summary>
        /// <param name="devicePtr"></param>
        public void FreeMemory(CUdeviceptr devicePtr)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemFree_v2(DriverAPINativeMethods.MemoryManagement.cuMemFree_v2), devicePtr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree", res));
            //res = DriverAPI.MemoryManagement.cuMemFree(dBuffer);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Returns the total device memory in bytes
        /// </summary>
        /// <returns></returns>
        public SizeT GetTotalDeviceMemorySize()
        {
            CUResult res;
            SizeT size = 0, free = 0;
            object[] paramMemGetInfo = { free, size };
            res = (CUResult)_dispatcher.Invoke(new cuMemGetInfo_v2(DriverAPINativeMethods.MemoryManagement.cuMemGetInfo_v2), paramMemGetInfo);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemGetInfo", res));
            //res = DriverAPI.MemoryManagement.cuMemGetInfo(ref free, ref size);
            if (res != CUResult.Success) throw new CudaException(res);
            size = (uint)paramMemGetInfo[1];
            return size;
        }

        /// <summary>
        /// Returns the free available device memory in bytes
        /// </summary>
        /// <returns></returns>
        public uint GetFreeDeviceMemorySize()
        {
            CUResult res;
            SizeT size = 0, free = 0;
            object[] paramMemGetInfo = { free, size };
            res = (CUResult)_dispatcher.Invoke(new cuMemGetInfo_v2(DriverAPINativeMethods.MemoryManagement.cuMemGetInfo_v2), paramMemGetInfo);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemGetInfo", res));
            //res = DriverAPI.MemoryManagement.cuMemGetInfo(ref free, ref size);
            if (res != CUResult.Success) throw new CudaException(res);
            size = (uint)paramMemGetInfo[0];
            return free;
        }        

        #region CopyToDevice
        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        /// <param name="sizeInBytes">Number of bytes to copy</param>
        public void CopyToDevice(CUdeviceptr dest, IntPtr source, SizeT sizeInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDIntPtr(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, sizeInBytes);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <typeparam name="T">T must be of value type, i.e. a struct</typeparam>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source pointer to host memory</param>
        public void CopyToDevice<T>(CUdeviceptr dest, T[] source) where T : struct
        {
            SizeT sizeInBytes =source.LongLength * Marshal.SizeOf(typeof(T));
            GCHandle handle = GCHandle.Alloc(source, GCHandleType.Pinned);
            IntPtr ptr = handle.AddrOfPinnedObject();
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDIntPtr(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, ptr, sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ptr, sizeInBytes);
            handle.Free();
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <typeparam name="T">T must be of value type, i.e. a struct</typeparam>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source pointer to host memory</param>
        public void CopyToDevice<T>(CUdeviceptr dest, T source) where T : struct
        {
            SizeT sizeInBytes = Marshal.SizeOf(typeof(T));
            GCHandle handle = GCHandle.Alloc(source, GCHandleType.Pinned);
            IntPtr ptr = handle.AddrOfPinnedObject();
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDIntPtr(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, ptr, sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ptr, sizeInBytes);
            handle.Free();
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, byte[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDByteA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, source.LongLength);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, source.LongLength);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, double[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDDoubleA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * sizeof(double)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * sizeof(double)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, float[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloatA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * sizeof(float)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * sizeof(float)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, int[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDIntA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * sizeof(int)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * sizeof(int)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, long[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDLongA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * sizeof(long)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * sizeof(long)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, sbyte[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDSByteA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * sizeof(sbyte)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * sizeof(sbyte)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, short[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDShortA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * sizeof(short)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * sizeof(short)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, uint[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUIntA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * sizeof(uint)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * sizeof(uint)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, ulong[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDULongA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * sizeof(ulong)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * sizeof(ulong)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, ushort[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUShortA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * sizeof(ushort)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * sizeof(ushort)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        #region VectorTypesArray
        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.dim3[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDDim3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.dim3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.dim3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.char1[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDChar1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.char1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.char1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.char2[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDChar2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.char2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.char2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.char3[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDChar3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.char3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.char3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.char4[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDChar4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.char4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.char4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.uchar1[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUChar1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.uchar2[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUChar2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.uchar3[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUChar3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.uchar4[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUChar4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        
        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.short1[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDShort1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.short1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.short1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.short2[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDShort2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.short2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.short2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.short3[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDShort3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.short3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.short3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.short4[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDShort4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.short4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.short4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.ushort1[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUShort1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.ushort2[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUShort2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.ushort3[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUShort4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.ushort4[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUShort4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.int1[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDInt1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.int1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.int1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.int2[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDInt2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.int2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.int2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.int3[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDInt3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.int3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.int3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.int4[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDInt4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.int4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.int4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.uint1[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUInt1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.uint2[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUInt2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.uint3[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUInt3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.uint4[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUInt4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.long1[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDLong1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.long1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.long1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.long2[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDLong2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.long2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.long2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.long3[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDLong3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.long3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.long3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.long4[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDLong4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.long4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.long4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.ulong1[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDULong1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.ulong2[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDULong2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.ulong3[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDULong3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.ulong4[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDULong4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.float1[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloat1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.float1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.float1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.float2[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloat2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.float2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.float2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.float3[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloat3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.float3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.float3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.float4[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloat4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.float4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.float4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.double1[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDDouble1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.double1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.double1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.double2[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDDouble2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.double2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.double2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.cuDoubleComplex[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDDoubleComplexA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuDoubleComplex))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuDoubleComplex))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.cuDoubleReal[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDDoubleRealA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuDoubleReal))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuDoubleReal))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.cuFloatComplex[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloatComplexA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuFloatComplex))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuFloatComplex))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source array</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.cuFloatReal[] source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloatRealA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuFloatReal))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuFloatReal))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        #endregion

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, byte source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDByte(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, sizeof(byte));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, sizeof(byte));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, double source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDDouble(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, sizeof(double));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, sizeof(double));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, float source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloat(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, sizeof(float));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, sizeof(float));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, int source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDInt(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, sizeof(int));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, sizeof(int));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, long source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDLong(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, sizeof(long));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, sizeof(long));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, sbyte source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDSByte(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, sizeof(sbyte));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, sizeof(sbyte));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, short source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDShort(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, sizeof(short));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, sizeof(short));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, uint source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUInt(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, sizeof(uint));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, sizeof(uint));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, ulong source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDULong(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, sizeof(ulong));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, sizeof(ulong));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, ushort source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUShort(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, sizeof(ushort));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, sizeof(ushort));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        #region VectorTypes
        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.dim3 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDDim3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.dim3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.dim3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.char1 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDChar1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.char1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.char1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.char2 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDChar2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.char2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.char2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.char3 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDChar3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.char3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.char3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.char4 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDChar4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.char4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.char4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.uchar1 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUChar1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.uchar1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.uchar1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.uchar2 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUChar2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.uchar2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.uchar2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.uchar3 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUChar3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.uchar3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.uchar3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.uchar4 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUChar4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.uchar4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.uchar4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.short1 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDShort1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.short1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.short1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.short2 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDShort2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.short2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.short2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.short3 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDShort3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.short3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.short3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.short4 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDShort4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.short4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.short4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.ushort1 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUShort1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.ushort1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.ushort1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.ushort2 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUShort2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.ushort2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.ushort2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.ushort3 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUShort3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.ushort3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.ushort3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.ushort4 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUShort4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.ushort4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.ushort4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.int1 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDInt1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.int1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.int1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.int2 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDInt2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.int2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.int2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.int3 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDInt3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.int3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.int3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.int4 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDInt4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.int4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.int4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.uint1 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUInt1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.uint1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.uint1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.uint2 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUInt2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.uint2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.uint2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.uint3 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUInt3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.uint3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.uint3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.uint4 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDUInt4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.uint4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.uint4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.long1 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDLong1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.long1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.long1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.long2 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDLong2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.long2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.long2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.long3 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDLong3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.long3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.long3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.long4 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDLong4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.long4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.long4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.ulong1 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDULong1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.ulong1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.ulong1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.ulong2 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDULong2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.ulong2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.ulong2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.ulong3 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDULong3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.ulong3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.ulong3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.ulong4 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDULong4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.ulong4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.ulong4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.float1 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloat1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.float1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.float1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.float2 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloat2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.float2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.float2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.float3 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloat3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.float3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.float3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.float4 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloat4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.float4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.float4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.double1 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDDouble1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.double1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.double1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.double2 source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDDouble2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.double2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.double2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.cuDoubleComplex source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDDoubleComplex(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.cuDoubleComplex)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.cuDoubleComplex)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.cuDoubleReal source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDDoubleReal(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.cuDoubleReal)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.cuDoubleReal)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.cuFloatComplex source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloatComplex(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.cuFloatComplex)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.cuFloatComplex)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to device memory
        /// </summary>
        /// <param name="dest">Destination CUdeviceptr (Pointer to device memory)</param>
        /// <param name="source">Source value</param>
        public void CopyToDevice(CUdeviceptr dest, VectorTypes.cuFloatReal source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoDFloatReal(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.cuFloatReal)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dest, ref source, Marshal.SizeOf(typeof(VectorTypes.cuFloatReal)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        #endregion
        #endregion

        #region CopyToHost

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <typeparam name="T">T must be of value type, i.e. a struct</typeparam>
        /// <param name="dest">Destination data in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost<T>(T[] dest, CUdeviceptr source) where T : struct
        {
            SizeT sizeInBytes = (dest.LongLength * Marshal.SizeOf(typeof(T)));
            CUResult res;
            GCHandle handle = GCHandle.Alloc(dest, GCHandleType.Pinned);
            IntPtr ptr = handle.AddrOfPinnedObject();
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHIntPtr(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), ptr, source, sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ptr, source, sizeInBytes);
            handle.Free();
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        
        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <typeparam name="T">T must be of value type, i.e. a struct</typeparam>
        /// <param name="dest">Destination data in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost<T>(T dest, CUdeviceptr source) where T : struct
        {
            SizeT sizeInBytes = Marshal.SizeOf(typeof(T));
            CUResult res;
            GCHandle handle = GCHandle.Alloc(dest, GCHandleType.Pinned);
            IntPtr ptr = handle.AddrOfPinnedObject();
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHIntPtr(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), ptr, source, sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ptr, source, sizeInBytes);
            handle.Free();
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(byte[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHByteA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * sizeof(byte));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * sizeof(byte));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(double[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHDoubleA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * sizeof(double));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * sizeof(double));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(float[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHFloatA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * sizeof(float));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * sizeof(float));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(int[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHIntA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * sizeof(int));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * sizeof(int));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination pointer to host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        /// <param name="sizeInBytes">Number of bytes to copy</param>
        public void CopyToHost(IntPtr dest, CUdeviceptr source, SizeT sizeInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHIntPtr(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, sizeInBytes);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(long[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHLongA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * sizeof(long));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * sizeof(long));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(sbyte[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHSByteA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * sizeof(sbyte));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * sizeof(sbyte));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(short[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHShortA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * sizeof(short));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * sizeof(short));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(uint[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUIntA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * sizeof(uint));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * sizeof(uint));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(ulong[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHULongA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * sizeof(ulong));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * sizeof(ulong));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(ushort[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUShortA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * sizeof(ushort));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * sizeof(ushort));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        #region VectorTypeArray
        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.dim3[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHDim3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.dim3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.dim3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.char1[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHChar1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.char1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.char1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.char2[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHChar2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.char2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.char2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.char3[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHChar3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.char3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.char3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.char4[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHChar4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.char4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.char4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.uchar1[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUChar1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.uchar2[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUChar2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.uchar3[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHChar3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.uchar4[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUChar4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.short1[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHShort1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.short1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.short1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.short2[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHShort2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source,dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.short2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.short2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.short3[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHShort3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.short3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.short3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.short4[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHShort4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.short4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.short4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.ushort1[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUShort1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.ushort2[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUShort2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.ushort3[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUShort3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.ushort4[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUShort4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.int1[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHInt1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.int1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.int1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.int2[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHInt2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.int2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.int2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.int3[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHInt3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.int3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.int3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.int4[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHInt4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.int4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.int4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.uint1[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUInt1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.uint2[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUInt2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.uint3[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUInt3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.uint4[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUInt4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.long1[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHLong1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.long1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.long1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.long2[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHLong2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.long2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.long2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.long3[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHLong3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.long3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.long3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.long4[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHLong4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.long4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.long4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.ulong1[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHULong1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.ulong2[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHULong2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.ulong3[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHULong3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.ulong4[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHULong4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.float1[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHFloat1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.float1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.float1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.float2[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHFloat2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.float2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.float2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.float3[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHFloat3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.float3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.float3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.float4[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHFloat4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.float4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.float4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.double1[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHDouble1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.double1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.double1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.double2[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHDouble2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.double2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.double2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.cuDoubleComplex[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHDoubleComplexA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuDoubleComplex)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuDoubleComplex)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.cuDoubleReal[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHDoubleRealA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuDoubleReal)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuDoubleReal)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.cuFloatComplex[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHFloatComplexA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuFloatComplex)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuFloatComplex)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination array in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.cuFloatReal[] dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHFloatRealA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuFloatReal)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, source, dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuFloatReal)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        #endregion

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(byte dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHByte(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, sizeof(ushort));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, sizeof(byte));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(double dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHDouble(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, sizeof(double));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, sizeof(double));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(float dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHFloat(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, sizeof(float));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, sizeof(float));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(int dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHInt(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, sizeof(int));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, sizeof(int));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(long dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHLong(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, sizeof(long));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, sizeof(long));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(sbyte dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHSByte(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, sizeof(sbyte));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, sizeof(sbyte));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(short dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHShort(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, sizeof(short));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, sizeof(short));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(uint dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUInt(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, sizeof(uint));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, sizeof(uint));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(ulong dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHULong(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, sizeof(ulong));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, sizeof(ulong));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(ushort dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUShort(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, sizeof(ushort));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, sizeof(ushort));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        #region VectorTypes
        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.dim3 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHDim3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.dim3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.dim3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.char1 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHChar1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.char1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.char1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.char2 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHChar2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.char2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.char2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.char3 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHChar3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.char3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.char3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.char4 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHChar4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.char4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.char4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.uchar1 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUChar1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.uchar1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.uchar1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.uchar2 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUChar2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.uchar2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.uchar2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.uchar3 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUChar3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.uchar3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.uchar3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.uchar4 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUChar4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.uchar4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.uchar4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.short1 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHShort1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.short1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.short1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.short2 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHShort2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.short2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.short2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.short3 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHShort3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.short3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.short3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.short4 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHShort4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.short4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.short4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.ushort1 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUShort1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.ushort1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.ushort1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.ushort2 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUShort2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.ushort2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.ushort2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.ushort3 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUShort3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.ushort3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.ushort3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.ushort4 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUShort4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.ushort4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.ushort4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.int1 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHInt1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.int1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.int1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.int2 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHInt2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.int2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.int2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.int3 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHInt3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.int3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.int3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.int4 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHInt4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.int4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.int4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.uint1 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUInt1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.uint1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.uint1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.uint2 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUInt2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.uint2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.uint2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.uint3 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUInt3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.uint3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.uint3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.uint4 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHUInt4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.uint4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.uint4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.long1 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHLong1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.long1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.long1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.long2 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHLong2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.long2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.long2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.long3 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHLong3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.long3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.long3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.long4 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHLong4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.long4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.long4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.ulong1 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHULong1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.ulong1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.ulong1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.ulong2 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHULong2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.ulong2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.ulong2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.ulong3 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHULong3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.ulong3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.ulong3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.ulong4 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHULong4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.ulong4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.ulong4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.float1 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHFloat1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.float1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.float1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.float2 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHFloat2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.float2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.float2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.float3 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHFloat3(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.float3)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.float3)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.float4 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHFloat4(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.float4)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.float4)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.double1 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHDouble1(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.double1)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.double1)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.double2 dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHDouble2(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.double2)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.double2)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.cuDoubleComplex dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHDoubleComplex(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.cuDoubleComplex)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.cuDoubleComplex)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.cuDoubleReal dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHDoubleReal(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.cuDoubleReal)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.cuDoubleReal)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.cuFloatComplex dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHFloatComplex(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.cuFloatComplex)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.cuFloatComplex)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to host memory
        /// </summary>
        /// <param name="dest">Destination value in host memory</param>
        /// <param name="source">Source CUdeviceptr (Pointer to device memory)</param>
        public void CopyToHost(VectorTypes.cuFloatReal dest, CUdeviceptr source)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoHFloatReal(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2), dest, source, Marshal.SizeOf(typeof(VectorTypes.cuFloatReal)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
            //res = DriverAPI.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ref dest, source, Marshal.SizeOf(typeof(VectorTypes.cuFloatReal)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        #endregion
        #endregion

        /// <summary>
        /// Retrieve device properties (Doesn't need to be dispatched)
        /// </summary>
        /// <returns>DeviceProperties</returns>
        public CudaDeviceProperties GetDeviceInfo()
        {
            CudaDeviceProperties props = new CudaDeviceProperties();
            byte[] devName = new byte[256];
            int major = 0, minor = 0;


            CUResult res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetName(devName, 256, _device);
            if (res != CUResult.Success) throw new CudaException(res);

            System.Text.ASCIIEncoding enc = new System.Text.ASCIIEncoding();
            props.DeviceName = enc.GetString(devName).Replace("\0", ""); 
          
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceComputeCapability(ref major, ref minor, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.ComputeCapability = new Version(major, minor);
          
            
            int driverVersion = 0;
            res = DriverAPINativeMethods.cuDriverGetVersion(ref driverVersion);
            if (res != CUResult.Success) throw new CudaException(res);
            props.DriverVersion = new Version(driverVersion/1000, driverVersion%100);



            SizeT totalGlobalMem = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceTotalMem_v2(ref totalGlobalMem, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.TotalGlobalMemory = totalGlobalMem;

            
            int multiProcessorCount = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref multiProcessorCount, CUDeviceAttribute.MultiProcessorCount, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.MultiProcessorCount = multiProcessorCount;


            int totalConstantMemory = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref totalConstantMemory, CUDeviceAttribute.TotalConstantMemory, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.TotalConstantMemory = totalConstantMemory;    

            int sharedMemPerBlock = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref sharedMemPerBlock, CUDeviceAttribute.MaxSharedMemoryPerBlock, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.SharedMemoryPerBlock = sharedMemPerBlock;    
            
            int regsPerBlock = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref regsPerBlock, CUDeviceAttribute.MaxRegistersPerBlock, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.RegistersPerBlock = regsPerBlock;    
            
            int warpSize = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref warpSize, CUDeviceAttribute.WarpSize, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.WarpSize = warpSize;    
            
            int maxThreadsPerBlock = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref maxThreadsPerBlock, CUDeviceAttribute.MaxThreadsPerBlock, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.MaxThreadsPerBlock = maxThreadsPerBlock;    
            
            ManagedCuda.VectorTypes.int3 blockDim = new VectorTypes.int3();
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref blockDim.x, CUDeviceAttribute.MaxBlockDimX, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref blockDim.y, CUDeviceAttribute.MaxBlockDimY, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref blockDim.z, CUDeviceAttribute.MaxBlockDimZ, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.MaxBlockDim = new VectorTypes.dim3((uint)blockDim.x, (uint)blockDim.y, (uint)blockDim.z);

            ManagedCuda.VectorTypes.int3 gridDim = new VectorTypes.int3();
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref gridDim.x, CUDeviceAttribute.MaxGridDimX, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref gridDim.y, CUDeviceAttribute.MaxGridDimY, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref gridDim.z, CUDeviceAttribute.MaxGridDimZ, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.MaxGridDim = new VectorTypes.dim3((uint)gridDim.x, (uint)gridDim.y, (uint)gridDim.z);    
            
            int memPitch = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref memPitch, CUDeviceAttribute.MaxPitch, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.MemoryPitch = memPitch;    
            
            int textureAlign = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref textureAlign, CUDeviceAttribute.TextureAlignment, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.TextureAlign = textureAlign;    
            
            int clockRate = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref clockRate, CUDeviceAttribute.ClockRate, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.ClockRate = clockRate;    
            


            int gpuOverlap = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref gpuOverlap, CUDeviceAttribute.GPUOverlap, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.GpuOverlap = gpuOverlap > 0;    
            

            int kernelExecTimeoutEnabled = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref kernelExecTimeoutEnabled, CUDeviceAttribute.KernelExecTimeout, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.KernelExecTimeoutEnabled = kernelExecTimeoutEnabled > 0;    
            
            int integrated = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref integrated, CUDeviceAttribute.Integrated, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.Integrated = integrated > 0;    
            
            int canMapHostMemory = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref canMapHostMemory, CUDeviceAttribute.CanMapHostMemory, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.CanMapHostMemory = canMapHostMemory > 0;    
            
            int computeMode = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref computeMode, CUDeviceAttribute.ComputeMode, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.ComputeMode = (BasicTypes.CUComputeMode)computeMode;

            int maxtexture1DWidth = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref maxtexture1DWidth, CUDeviceAttribute.MaximumTexture1DWidth, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.MaximumTexture1DWidth = maxtexture1DWidth;

            int maxtexture2DWidth = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref maxtexture2DWidth, CUDeviceAttribute.MaximumTexture2DWidth, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.MaximumTexture2DWidth = maxtexture2DWidth;

            int maxtexture2DHeight = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref maxtexture2DHeight, CUDeviceAttribute.MaximumTexture2DHeight, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.MaximumTexture2DHeight = maxtexture2DHeight;

            int maxtexture3DWidth = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref maxtexture3DWidth, CUDeviceAttribute.MaximumTexture3DWidth, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.MaximumTexture3DWidth = maxtexture2DWidth;

            int maxtexture3DHeight = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref maxtexture3DHeight, CUDeviceAttribute.MaximumTexture3DHeight, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.MaximumTexture3DHeight = maxtexture2DHeight;

            int maxtexture3DDepth = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref maxtexture3DDepth, CUDeviceAttribute.MaximumTexture3DDepth, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.MaximumTexture3DDepth = maxtexture2DHeight;

            int maxtexture2DArray_Width = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref maxtexture2DArray_Width, CUDeviceAttribute.MaximumTexture2DArray_Width, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.MaximumTexture2DArrayWidth = maxtexture2DArray_Width;

            int maxtexture2DArray_Height = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref maxtexture2DArray_Height, CUDeviceAttribute.MaximumTexture2DArray_Height, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.MaximumTexture2DArrayHeight = maxtexture2DArray_Height;

            int maxtexture2DArray_NumSlices = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref maxtexture2DArray_NumSlices, CUDeviceAttribute.MaximumTexture2DArray_NumSlices, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.MaximumTexture2DArrayNumSlices = maxtexture2DArray_NumSlices;

            int surfaceAllignment = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref surfaceAllignment, CUDeviceAttribute.SurfaceAllignment, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.SurfaceAllignment = surfaceAllignment;

            int concurrentKernels = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref concurrentKernels, CUDeviceAttribute.ConcurrentKernels, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.ConcurrentKernels = concurrentKernels > 0;

            int ECCEnabled = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref ECCEnabled, CUDeviceAttribute.ECCEnabled, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.EccEnabled = ECCEnabled > 0;

            int PCIBusID = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref PCIBusID, CUDeviceAttribute.PCIBusID, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.PciBusId = PCIBusID;

            int PCIDeviceID = 0;
            res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetAttribute(ref PCIDeviceID, CUDeviceAttribute.PCIDeviceID, _device);
            if (res != CUResult.Success) throw new CudaException(res);
            props.PciDeviceId = PCIDeviceID;   


            return props;
        }
        #endregion

        #region Properties
        /// <summary>
        /// Gets the Cuda context bound to this managed Cuda object
        /// </summary>
        public CUcontext Context
        {
            get { return _context; }
        }

        /// <summary>
        /// Gets the Cuda device allocated to the Cuda Context
        /// </summary>
        public CUdevice Device
        {
            get { return _device; }
        }

        /// <summary>
        /// Gets the Id of the Cuda device.
        /// </summary>
        public int DeviceId
        {
            get { return _deviceID; }
        }

        /// <summary>
        /// Returns the thread assigned to this Cuda object
        /// </summary>
        public Thread Thread
        {
            get { return _thread; }
        }
        #endregion
    }
}