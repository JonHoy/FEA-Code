using System;
using System.Collections.Generic;
using System.Text;
using ManagedCuda.BasicTypes;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace ManagedCuda.Dispatched
{
    /// <summary>
    /// CudaLinearTexture2D
    /// </summary>
    [Obsolete("CUDA 4.0 API calls are now thread safe. Dispatched... wrapper classes are of no use now.")]
    public class DispatchedCudaTextureLinearPitched2D<T> : DispatchedCudaBaseClass, IDisposable where T : struct
    {
        CUtexref _texref;
        CUFilterMode _filtermode;
        CUTexRefSetFlags _flags;
        CUAddressMode _addressMode0, _addressMode1;
        CUArrayFormat _format;
        SizeT _height;
        SizeT _width;
        uint _channelSize;
        SizeT _dataSize;
        int _numChannels;
        string _name;
        CUmodule _module;
        CUfunction _cufunction;
        DispatchedCudaPitchedDeviceVariable<T> _devVar;
        CUResult res;
        bool disposed;

        #region Constructors
        /// <summary>
        /// Creates a new 2D texture from linear memory. Allocates a new device variable
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="texName"></param>
        /// <param name="addressMode"></param>
        /// <param name="filterMode"></param>
        /// <param name="flags"></param>
        /// <param name="format"></param>
        /// <param name="height">In elements</param>
        /// <param name="width">In elements</param>
        public DispatchedCudaTextureLinearPitched2D(DispatchedCudaKernel kernel, string texName, CUAddressMode addressMode, CUFilterMode filterMode, CUTexRefSetFlags flags, CUArrayFormat format, SizeT height, SizeT width)
            : this(kernel, texName, addressMode, addressMode, filterMode, flags, format, height, width)
        {

        }

        /// <summary>
        /// Creates a new 2D texture from linear memory. Allocates a new device variable
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="texName"></param>
        /// <param name="addressMode0"></param>
        /// <param name="addressMode1"></param>
        /// <param name="filterMode"></param>
        /// <param name="flags"></param>
        /// <param name="format"></param>
        /// <param name="height">In elements</param>
        /// <param name="width">In elements</param>
        public DispatchedCudaTextureLinearPitched2D(DispatchedCudaKernel kernel, string texName, CUAddressMode addressMode0, CUAddressMode addressMode1, CUFilterMode filterMode, CUTexRefSetFlags flags, CUArrayFormat format, SizeT height, SizeT width)
            : base(kernel.Dispatcher)
        {
            _texref = new CUtexref();
            object[] paramModuleGetTexRef = { _texref, kernel.CUModule, texName };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetTexRef(DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef), paramModuleGetTexRef);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Texture name: {3}", DateTime.Now, "cuModuleGetTexRef", res, texName));
            //res = DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef(ref _texref, module, texName);
            if (res != CUResult.Success) throw new CudaException(res);
            _texref = (CUtexref)paramModuleGetTexRef[0];

            res = (CUResult)_dispatcher.Invoke(new cuTexRefSetAddressMode(DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode), _texref, 0, addressMode0);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
            //res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(_texref, 0, addressMode0);
            if (res != CUResult.Success) throw new CudaException(res);

            res = (CUResult)_dispatcher.Invoke(new cuTexRefSetAddressMode(DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode), _texref, 1, addressMode1);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
            //res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(_texref, 1, addressMode1);
            if (res != CUResult.Success) throw new CudaException(res);

            res = (CUResult)_dispatcher.Invoke(new cuTexRefSetFilterMode(DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode), _texref, filterMode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
            //res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(_texref, filterMode);
            if (res != CUResult.Success) throw new CudaException(res);

            res = (CUResult)_dispatcher.Invoke(new cuTexRefSetFlags(DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags), _texref, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
            //res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(_texref, flags);
            if (res != CUResult.Success) throw new CudaException(res);

            _numChannels = CudaHelperMethods.GetNumChannels(typeof(T));
            res = (CUResult)_dispatcher.Invoke(new cuTexRefSetFormat(DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat), _texref, format, _numChannels);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
            //res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(_texref, format, _numChannels);
            if (res != CUResult.Success) throw new CudaException(res);

            _filtermode = filterMode;
            _flags = flags;
            _addressMode0 = addressMode0;
            _addressMode1 = addressMode1;
            _format = format;
            _height = height;
            _width = width;
            _name = texName;
            _module = kernel.CUModule;
            _cufunction = kernel.CUFunction;

            _channelSize = CudaHelperMethods.GetChannelSize(format);
            _dataSize = height * width * _numChannels * _channelSize;
            _devVar = new DispatchedCudaPitchedDeviceVariable<T>(height, width, kernel.Dispatcher);

            CUDAArrayDescriptor arrayDescr = new CUDAArrayDescriptor();
            arrayDescr.Format = format;
            arrayDescr.Height = height;
            arrayDescr.NumChannels = (uint)_numChannels;
            arrayDescr.Width = width;

            object[] paramTexRefSetAddress2D = {_texref, arrayDescr, _devVar.DevicePointer, _devVar.Pitch };

            res = (CUResult)_dispatcher.Invoke(new cuTexRefSetAddress2D_v2(DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddress2D_v2), paramTexRefSetAddress2D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddress2D", res));
            //res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddress2D(_texref, ref arrayDescr, _devVar.DevicePointer, _devVar.Pitch);
            if (res != CUResult.Success) throw new CudaException(res);

            //res = (CUResult)_dispatcher.Invoke(new cuParamSetTexRef(DriverAPINativeMethods.ParameterManagement.cuParamSetTexRef), kernel.CUFunction, CUParameterTexRef.Default, _texref);
            //Debug.WriteLine("{0:G}, {1}: {2}", DateTime.Now, "cuParamSetTexRef", res);
            ////res = DriverAPINativeMethods.ParameterManagement.cuParamSetTexRef(cufunction, CUParameterTexRef.Default, _texref);
            //if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Creates a new 2D texture from linear memory.
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="texName"></param>
        /// <param name="addressMode"></param>
        /// <param name="filterMode"></param>
        /// <param name="flags"></param>
        /// <param name="format"></param>
        /// <param name="deviceVar"></param>
        public DispatchedCudaTextureLinearPitched2D(DispatchedCudaKernel kernel, string texName, CUAddressMode addressMode, CUFilterMode filterMode, CUTexRefSetFlags flags, CUArrayFormat format, DispatchedCudaPitchedDeviceVariable<T> deviceVar)
            : this(kernel, texName, addressMode, addressMode, filterMode, flags, format, deviceVar)
        {

        }

        /// <summary>
        /// Creates a new 2D texture from linear memory.
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="texName"></param>
        /// <param name="addressMode0"></param>
        /// <param name="addressMode1"></param>
        /// <param name="filterMode"></param>
        /// <param name="flags"></param>
        /// <param name="format"></param>
        /// <param name="deviceVar"></param>
        public DispatchedCudaTextureLinearPitched2D(DispatchedCudaKernel kernel, string texName, CUAddressMode addressMode0, CUAddressMode addressMode1, CUFilterMode filterMode, CUTexRefSetFlags flags, CUArrayFormat format, DispatchedCudaPitchedDeviceVariable<T> deviceVar)
            : base(kernel.Dispatcher)
        {
            _texref = new CUtexref();
            object[] paramModuleGetTexRef = { _texref, kernel.CUModule, texName };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetTexRef(DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef), paramModuleGetTexRef);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Texture name: {3}", DateTime.Now, "cuModuleGetTexRef", res, texName));
            //res = DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef(ref _texref, module, texName);
            if (res != CUResult.Success) throw new CudaException(res);
            _texref = (CUtexref)paramModuleGetTexRef[0];

            res = (CUResult)_dispatcher.Invoke(new cuTexRefSetAddressMode(DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode), _texref, 0, addressMode0);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
            //res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(_texref, 0, addressMode0);
            if (res != CUResult.Success) throw new CudaException(res);

            res = (CUResult)_dispatcher.Invoke(new cuTexRefSetAddressMode(DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode), _texref, 1, addressMode1);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
            //res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(_texref, 1, addressMode1);
            if (res != CUResult.Success) throw new CudaException(res);

            res = (CUResult)_dispatcher.Invoke(new cuTexRefSetFilterMode(DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode), _texref, filterMode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
            //res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(_texref, filterMode);
            if (res != CUResult.Success) throw new CudaException(res);

            res = (CUResult)_dispatcher.Invoke(new cuTexRefSetFlags(DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags), _texref, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
            //res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(_texref, flags);
            if (res != CUResult.Success) throw new CudaException(res);

            _numChannels = CudaHelperMethods.GetNumChannels(typeof(T));
            res = (CUResult)_dispatcher.Invoke(new cuTexRefSetFormat(DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat), _texref, format, _numChannels);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
            //res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(_texref, format, _numChannels);
            if (res != CUResult.Success) throw new CudaException(res);

            _filtermode = filterMode;
            _flags = flags;
            _addressMode0 = addressMode0;
            _addressMode1 = addressMode1;
            _format = format;
            _height = deviceVar.Height;
            _width = deviceVar.Width;
            _name = texName;
            _module = kernel.CUModule;
            _cufunction = kernel.CUFunction;

            _channelSize = CudaHelperMethods.GetChannelSize(format);
            _dataSize = deviceVar.TotalSizeInBytes;
            _devVar = deviceVar;

            CUDAArrayDescriptor arrayDescr = new CUDAArrayDescriptor();
            arrayDescr.Format = format;
            arrayDescr.Height = _height;
            arrayDescr.NumChannels = (uint)_numChannels;
            arrayDescr.Width = _width;

            object[] paramTexRefSetAddress2D = { _texref, arrayDescr, _devVar.DevicePointer, _devVar.Pitch };

            res = (CUResult)_dispatcher.Invoke(new cuTexRefSetAddress2D_v2(DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddress2D_v2), paramTexRefSetAddress2D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddress2D", res));
            //res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddress2D(_texref, ref arrayDescr, _devVar.DevicePointer, _devVar.Pitch);
            if (res != CUResult.Success) throw new CudaException(res);

            //res = (CUResult)_dispatcher.Invoke(new cuParamSetTexRef(DriverAPINativeMethods.ParameterManagement.cuParamSetTexRef), kernel.CUFunction, CUParameterTexRef.Default, _texref);
            //Debug.WriteLine("{0:G}, {1}: {2}", DateTime.Now, "cuParamSetTexRef", res);
            ////res = DriverAPINativeMethods.ParameterManagement.cuParamSetTexRef(cufunction, CUParameterTexRef.Default, _texref);
            //if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~DispatchedCudaTextureLinearPitched2D()
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
                _devVar.Dispose();
                disposed = true;
                // the _texref reference is not destroyed explicitly, as it is done automatically when module is unloaded
            }
        }
        #endregion

        #region Properties
        /// <summary>
        /// TextureReference
        /// </summary>
        public CUtexref TextureReference
        {
            get { return _texref; }
        }

        /// <summary>
        /// Flags
        /// </summary>
        public CUTexRefSetFlags Flags
        {
            get { return _flags; }
        }

        /// <summary>
        /// AddressMode
        /// </summary>
        public CUAddressMode AddressMode0
        {
            get { return _addressMode0; }
        }

        /// <summary>
        /// AddressMode
        /// </summary>
        public CUAddressMode AddressMode1
        {
            get { return _addressMode1; }
        }

        /// <summary>
        /// Format
        /// </summary>
        public CUArrayFormat Format
        {
            get { return _format; }
        }

        /// <summary>
        /// Format
        /// </summary>
        public CUFilterMode Filtermode
        {
            get { return _filtermode; }
        }

        /// <summary>
        /// Height
        /// </summary>
        public SizeT Height
        {
            get { return _height; }
        }

        /// <summary>
        /// Width
        /// </summary>
        public SizeT Width
        {
            get { return _width; }
        }

        /// <summary>
        /// ChannelSize
        /// </summary>
        public uint ChannelSize
        {
            get { return _channelSize; }
        }

        /// <summary>
        /// TotalSizeInBytes
        /// </summary>
        public SizeT TotalSizeInBytes
        {
            get { return _dataSize; }
        }

        /// <summary>
        /// NumChannels
        /// </summary>
        public int NumChannels
        {
            get { return _numChannels; }
        }

        /// <summary>
        /// Name
        /// </summary>
        public string Name
        {
            get { return _name; }
        }

        /// <summary>
        /// Module
        /// </summary>
        public CUmodule Module
        {
            get { return _module; }
        }

        /// <summary>
        /// CUFunction
        /// </summary>
        public CUfunction CUFuntion
        {
            get { return _cufunction; }
        }

        /// <summary>
        /// Device variable in linear Memory
        /// </summary>
        public DispatchedCudaPitchedDeviceVariable<T> DeviceVar
        {
            get { return _devVar; }
        }
        #endregion
    }

}

