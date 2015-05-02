using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;
using CUDA.BasicTypes;
using CUDA.VectorTypes;

namespace CUDA.Dispatched
{
    /// <summary>
    /// Direct3D 9 Interoperability
    /// </summary>
    public sealed class DirectX9
    {
        internal const string CUDA_DRIVER_API_DLL_NAME = "nvcuda";
        internal const string CUDA1_OBSOLETE = "CUDA 1.x compatibility API. These functions are deprecated, please use the CUDA3 ones.";
        internal const string CUDA2_OBSOLETE = "CUDA 2.x compatibility API. These functions are deprecated, please use the CUDA3 ones.";

        /// <summary>
        /// Direct3D9 Interoperability for CUDA 3.x
        /// </summary>
        public sealed class CUDA3
        {
            /// <summary>
            /// Returns in <c>pCudaDevice</c> the CUDA-compatible device corresponding to the adapter name <c>pszAdapterName</c>
            /// obtained from <c>EnumDisplayDevices()</c> or <c>IDirect3D9::GetAdapterIdentifier()</c>.
            /// If no device on the adapter with name <c>pszAdapterName</c> is CUDA-compatible, then the call will fail.
            /// </summary>
            /// <param name="pCudaDevice">Returned CUDA device corresponding to pszAdapterName</param>
            /// <param name="pszAdapterName">Adapter name to query for device</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D9GetDevice(ref CUdevice pCudaDevice, string pszAdapterName);
            /// <summary>
            /// Creates a new CUDA context, enables interoperability for that context with the Direct3D device <c>pD3DDevice</c>, and
            /// associates the created CUDA context with the calling thread. The created <see cref="CUcontext"/> will be returned in <c>pCtx</c>.
            /// Direct3D resources from this device may be registered and mapped through the lifetime of this CUDA context.
            /// If <c>pCudaDevice</c> is non-NULL then the <see cref="CUdevice"/> on which this CUDA context was created will be returned in
            /// <c>pCudaDevice</c>.
            /// On success, this call will increase the internal reference count on <c>pD3DDevice</c>. This reference count will be decremented
            /// upon destruction of this context through <see cref="CUDA.DriverAPI.ContextManagement.cuCtxDestroy"/>. This context will cease to function if <c>pD3DDevice</c>
            /// is destroyed or encounters an error.
            /// </summary>
            /// <param name="pCtx">Returned newly created CUDA context</param>
            /// <param name="pCudaDevice">Returned pointer to the device on which the context was created</param>
            /// <param name="Flags">Context creation flags (see <see cref="CUDA.DriverAPI.ContextManagement.cuCtxCreate"/> for details)</param>
            /// <param name="pD3DDevice">Direct3D device to create interoperability context with</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorOutOfMemory"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D9CtxCreate(ref CUcontext pCtx, ref CUdevice pCudaDevice, CUCtxFlags Flags, IntPtr pD3DDevice);
            /// <summary>
            /// Registers the Direct3D 9 resource <c>pD3DResource</c> for access by CUDA and returns a CUDA handle to
            /// <c>pD3Dresource</c> in <c>pCudaResource</c>. The handle returned in <c>pCudaResource</c> may be used to map and
            /// unmap this resource until it is unregistered. On success this call will increase the internal reference count on
            /// <c>pD3DResource</c>. This reference count will be decremented when this resource is unregistered through <see cref="CUDA.DriverAPI.GraphicsInterop.cuGraphicsUnregisterResource"/>.<para/>
            /// This call is potentially high-overhead and should not be called every frame in interactive applications.<para/>
            /// The type of pD3DResource must be one of the following:
            /// <list type="table">  
            /// <listheader><term>Type of <c>pD3DResource</c></term><description>Restriction</description></listheader>  
            /// <item><term>IDirect3DVertexBuffer9</term><description>
            /// May be accessed through a device pointer.
            /// </description></item>  
            /// <item><term>IDirect3DIndexBuffer9</term><description>
            /// May be accessed through a device pointer.
            /// </description></item>  
            /// <item><term>IDirect3DSurface9</term><description>
            /// May be accessed through an array. Only stand-alone objects of type <c>IDirect3DSurface9</c>
            /// may be explicitly shared. In particular, individual mipmap levels and faces of cube maps may not be registered
            /// directly. To access individual surfaces associated with a texture, one must register the base texture object.
            /// </description></item>  
            /// <item><term>IDirect3DBaseTexture9</term><description>
            /// Individual surfaces on this texture may be accessed through an array.
            /// </description></item> 
            /// </list> 
            /// The Flags argument may be used to specify additional parameters at register time. The only valid value for this
            /// parameter is <see cref="CUGraphicsRegisterFlags.None"/>. <para/>
            /// Not all Direct3D resources of the above types may be used for interoperability with CUDA. The following are some
            /// limitations.<param/>
            /// • The primary rendertarget may not be registered with CUDA.<param/>
            /// • Resources allocated as shared may not be registered with CUDA.<param/>
            /// • Textures which are not of a format which is 1, 2, or 4 channels of 8, 16, or 32-bit integer or floating-point data
            /// cannot be shared.<param/>
            /// • Surfaces of depth or stencil formats cannot be shared.<param/>
            /// If Direct3D interoperability is not initialized for this context using <see cref="cuD3D9CtxCreate"/> then
            /// <see cref="CUResult.ErrorInvalidContext"/> is returned. If <c>pD3DResource</c> is of incorrect type or is already registered then
            /// <see cref="CUResult.ErrorInvalidHandle"/> is returned. If <c>pD3DResource</c> cannot be registered then 
            /// <see cref="CUResult.ErrorUnknown"/> is returned. If <c>Flags</c> is not one of the above specified value then <see cref="CUResult.ErrorInvalidValue"/>
            /// is returned.
            /// </summary>
            /// <param name="pCudaResource">Returned graphics resource handle</param>
            /// <param name="pD3DResource">Direct3D resource to register</param>
            /// <param name="Flags">Parameters for resource registration</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidHandle"/>, <see cref="CUResult.ErrorOutOfMemory"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuGraphicsD3D9RegisterResource(ref CUgraphicsResource pCudaResource, IntPtr pD3DResource, CUGraphicsRegisterFlags Flags);
        }

        /// <summary>
        /// Direct3D9 Interoperability for CUDA 2.x
        /// </summary>
        [Obsolete(CUDA2_OBSOLETE)]
        public sealed class CUDA2
        {
            /// <summary>
            /// Flags to register a resource
            /// </summary>
            [Obsolete(CUDA2_OBSOLETE)]
            public enum CUD3D9RegisterFlags
            {
                /// <summary>
                /// 
                /// </summary>
                NONE = 0x00,
                /// <summary>
                /// 
                /// </summary>
                ARRAY = 0x01
            }

            
            /// <summary>
            /// Flags to map or unmap a resource
            /// </summary>
            [Obsolete(CUDA2_OBSOLETE)]
            public enum CUD3D9MapFlags
            {
                /// <summary>
                /// 
                /// </summary>
                NONE = 0x00,
                /// <summary>
                /// 
                /// </summary>
                READONLY = 0x01,
                /// <summary>
                /// 
                /// </summary>
                WRITEDISCARD = 0x02
            }

            /// <summary>
            /// 
            /// </summary>
            /// <param name="ppD3DDevice"></param>
            /// <returns></returns>
            [Obsolete(CUDA2_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D9GetDirect3DDevice(ref IntPtr ppD3DDevice);

            /// <summary>
            /// 
            /// </summary>
            /// <param name="pResource"></param>
            /// <param name="Flags"></param>
            /// <returns></returns>
            [Obsolete(CUDA2_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D9RegisterResource(IntPtr pResource, CUD3D9RegisterFlags Flags);
            /// <summary>
            /// 
            /// </summary>
            /// <param name="pResource"></param>
            /// <returns></returns>
            [Obsolete(CUDA2_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D9UnregisterResource(IntPtr pResource);

            /// <summary>
            /// 
            /// </summary>
            /// <param name="count"></param>
            /// <param name="ppResource"></param>
            /// <returns></returns>
            [Obsolete(CUDA2_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D9MapResources(uint count, ref IntPtr ppResource);
            /// <summary>
            /// 
            /// </summary>
            /// <param name="count"></param>
            /// <param name="ppResource"></param>
            /// <returns></returns>
            [Obsolete(CUDA2_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D9UnmapResources(uint count, ref IntPtr ppResource);
            /// <summary>
            /// 
            /// </summary>
            /// <param name="pResource"></param>
            /// <param name="Flags"></param>
            /// <returns></returns>
            [Obsolete(CUDA2_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D9ResourceSetMapFlags(IntPtr pResource, CUD3D9MapFlags Flags);

            /// <summary>
            /// 
            /// </summary>
            /// <param name="pWidth"></param>
            /// <param name="pHeight"></param>
            /// <param name="pDepth"></param>
            /// <param name="pResource"></param>
            /// <param name="Face"></param>
            /// <param name="Level"></param>
            /// <returns></returns>
            [Obsolete(CUDA2_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D9ResourceGetSurfaceDimensions(ref uint pWidth, ref uint pHeight, ref uint pDepth, IntPtr pResource, uint Face, uint Level);
            /// <summary>
            /// 
            /// </summary>
            /// <param name="pArray"></param>
            /// <param name="pResource"></param>
            /// <param name="Face"></param>
            /// <param name="Level"></param>
            /// <returns></returns>
            [Obsolete(CUDA2_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D9ResourceGetMappedArray(ref CUarray pArray, IntPtr pResource, uint Face, uint Level);
            /// <summary>
            /// 
            /// </summary>
            /// <param name="pDevPtr"></param>
            /// <param name="pResource"></param>
            /// <param name="Face"></param>
            /// <param name="Level"></param>
            /// <returns></returns>
            [Obsolete(CUDA2_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D9ResourceGetMappedPointer(ref CUdeviceptr pDevPtr, IntPtr pResource, uint Face, uint Level);
            /// <summary>
            /// 
            /// </summary>
            /// <param name="pSize"></param>
            /// <param name="pResource"></param>
            /// <param name="Face"></param>
            /// <param name="Level"></param>
            /// <returns></returns>
            [Obsolete(CUDA2_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D9ResourceGetMappedSize(ref uint pSize, IntPtr pResource, uint Face, uint Level);
            /// <summary>
            /// 
            /// </summary>
            /// <param name="pPitch"></param>
            /// <param name="pPitchSlice"></param>
            /// <param name="pResource"></param>
            /// <param name="Face"></param>
            /// <param name="Level"></param>
            /// <returns></returns>
            [Obsolete(CUDA2_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D9ResourceGetMappedPitch(ref uint pPitch, ref uint pPitchSlice, IntPtr pResource, uint Face, uint Level);
        }

        /// <summary>
        /// Direct3D9 Interoperability for CUDA 1.x
        /// </summary>
        [Obsolete(CUDA1_OBSOLETE)]
        public sealed class CUDA1
        {
            /// <summary>
            /// 
            /// </summary>
            /// <param name="pDevice"></param>
            /// <returns></returns>
            [Obsolete(CUDA1_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D9Begin(IntPtr pDevice);
            /// <summary>
            /// 
            /// </summary>
            /// <returns></returns>
            [Obsolete(CUDA1_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D9End();
            /// <summary>
            /// 
            /// </summary>
            /// <param name="pVB"></param>
            /// <returns></returns>
            [Obsolete(CUDA1_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D9RegisterVertexBuffer(IntPtr pVB);
            /// <summary>
            /// 
            /// </summary>
            /// <param name="pDevPtr"></param>
            /// <param name="pSize"></param>
            /// <param name="pVB"></param>
            /// <returns></returns>
            [Obsolete(CUDA1_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D9MapVertexBuffer(ref CUdeviceptr pDevPtr, ref uint pSize, IntPtr pVB);
            /// <summary>
            /// 
            /// </summary>
            /// <param name="pVB"></param>
            /// <returns></returns>
            [Obsolete(CUDA1_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D9UnmapVertexBuffer(IntPtr pVB);
            /// <summary>
            /// 
            /// </summary>
            /// <param name="pVB"></param>
            /// <returns></returns>
            [Obsolete(CUDA1_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D9UnregisterVertexBuffer(IntPtr pVB);
        }
    }

    /// <summary>
    /// Direct3D 10 Interoperability
    /// </summary>
    public sealed class DirectX10
    {
        internal const string CUDA_DRIVER_API_DLL_NAME = "nvcuda";
        internal const string CUDA2_OBSOLETE = "CUDA 2.x compatibility API. These functions are deprecated, please use the CUDA3 ones.";

        /// <summary>
        /// Direct3D10 Interoperability for CUDA 3.x
        /// </summary>
        public sealed class CUDA3
        {
            /// <summary>
            /// Returns in <c>device</c> the CUDA-compatible device corresponding to the adapter <c>pAdapter</c> obtained from 
            /// <c>IDXGIFactory::EnumAdapters</c>. This call will succeed only if a device on adapter <c>pAdapter</c> is Cuda-compatible.
            /// </summary>
            /// <param name="device">Returned CUDA device corresponding to pszAdapterName</param>
            /// <param name="pAdapter">Adapter (type: IDXGIAdapter)</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D10GetDevice(ref CUdevice device, IntPtr pAdapter);
            /// <summary>
            /// Creates a new CUDA context, enables interoperability for that context with the Direct3D device <c>pD3DDevice</c>, and
            /// associates the created CUDA context with the calling thread. The created <see cref="CUcontext"/> will be returned in <c>pCtx</c>.
            /// Direct3D resources from this device may be registered and mapped through the lifetime of this CUDA context.
            /// If <c>pCudaDevice</c> is non-NULL then the <see cref="CUdevice"/> on which this CUDA context was created will be returned in
            /// <c>pCudaDevice</c>.
            /// On success, this call will increase the internal reference count on <c>pD3DDevice</c>. This reference count will be decremented
            /// upon destruction of this context through <see cref="CUDA.DriverAPI.ContextManagement.cuCtxDestroy"/>. This context will cease to function if <c>pD3DDevice</c>
            /// is destroyed or encounters an error.
            /// </summary>
            /// <param name="pCtx">Returned newly created CUDA context</param>
            /// <param name="pCudaDevice">Returned pointer to the device on which the context was created</param>
            /// <param name="Flags">Context creation flags (see <see cref="CUDA.DriverAPI.ContextManagement.cuCtxCreate"/> for details)</param>
            /// <param name="pD3DDevice">Direct3D device to create interoperability context with</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorOutOfMemory"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D10CtxCreate(ref CUcontext pCtx, ref CUdevice pCudaDevice, CUCtxFlags Flags, IntPtr pD3DDevice);
            /// <summary>
            /// Registers the Direct3D 10 resource <c>pD3DResource</c> for access by CUDA and returns a CUDA handle to
            /// <c>pD3Dresource</c> in <c>pCudaResource</c>. The handle returned in <c>pCudaResource</c> may be used to map and
            /// unmap this resource until it is unregistered. On success this call will increase the internal reference count on
            /// <c>pD3DResource</c>. This reference count will be decremented when this resource is unregistered through <see cref="CUDA.DriverAPI.GraphicsInterop.cuGraphicsUnregisterResource"/>.<para/>
            /// This call is potentially high-overhead and should not be called every frame in interactive applications.<para/>
            /// The type of pD3DResource must be one of the following:
            /// <list type="table">  
            /// <listheader><term>Type of <c>pD3DResource</c></term><description>Restriction</description></listheader>  
            /// <item><term>ID3D10Buffer</term><description>
            /// May be accessed through a device pointer.
            /// </description></item>  
            /// <item><term>ID3D10Texture1D</term><description>
            /// Individual subresources of the texture may be accessed via arrays.
            /// </description></item>  
            /// <item><term>ID3D10Texture2D</term><description>
            /// Individual subresources of the texture may be accessed via arrays.
            /// </description></item> 
            /// <item><term>ID3D10Texture3D</term><description>
            /// Individual subresources of the texture may be accessed via arrays.
            /// </description></item>  
            /// </list> 
            /// The Flags argument may be used to specify additional parameters at register time. The only valid value for this
            /// parameter is <see cref="CUGraphicsRegisterFlags.None"/>. <para/>
            /// Not all Direct3D resources of the above types may be used for interoperability with CUDA. The following are some
            /// limitations.<param/>
            /// • The primary rendertarget may not be registered with CUDA.<param/>
            /// • Resources allocated as shared may not be registered with CUDA.<param/>
            /// • Textures which are not of a format which is 1, 2, or 4 channels of 8, 16, or 32-bit integer or floating-point data
            /// cannot be shared.<param/>
            /// • Surfaces of depth or stencil formats cannot be shared.<param/>
            /// If Direct3D interoperability is not initialized for this context using <see cref="cuD3D10CtxCreate"/> then
            /// <see cref="CUResult.ErrorInvalidContext"/> is returned. If <c>pD3DResource</c> is of incorrect type or is already registered then
            /// <see cref="CUResult.ErrorInvalidHandle"/> is returned. If <c>pD3DResource</c> cannot be registered then 
            /// <see cref="CUResult.ErrorUnknown"/> is returned. If <c>Flags</c> is not one of the above specified value then <see cref="CUResult.ErrorInvalidValue"/>
            /// is returned.
            /// </summary>
            /// <param name="pCudaResource">Returned graphics resource handle</param>
            /// <param name="pD3DResource">Direct3D resource to register</param>
            /// <param name="Flags">Parameters for resource registration</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidHandle"/>, <see cref="CUResult.ErrorOutOfMemory"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuGraphicsD3D10RegisterResource(ref CUgraphicsResource pCudaResource, IntPtr pD3DResource, CUGraphicsRegisterFlags Flags);
        }

        /// <summary>
        /// Direct3D10 Interoperability for CUDA 2.x
        /// </summary>
        [Obsolete(CUDA2_OBSOLETE)]
        public sealed class CUDA2
        {
            /// <summary>
            /// Flags to register a resource
            /// </summary>
            [Obsolete(CUDA2_OBSOLETE)]
            public enum CUD3D10RegisterFlags
            {
                /// <summary>
                /// 
                /// </summary>
                NONE = 0x00,
                /// <summary>
                /// 
                /// </summary>
                ARRAY = 0x01
            }


            /// <summary>
            /// Flags to map or unmap a resource
            /// </summary>
            [Obsolete(CUDA2_OBSOLETE)]
            public enum CUD3D10MapFlags
            {
                /// <summary>
                /// 
                /// </summary>
                NONE = 0x00,
                /// <summary>
                /// 
                /// </summary>
                READONLY = 0x01,
                /// <summary>
                /// 
                /// </summary>
                WRITEDISCARD = 0x02
            }

            /// <summary>
            /// 
            /// </summary>
            /// <param name="pResource"></param>
            /// <param name="Flags"></param>
            /// <returns></returns>
            [Obsolete(CUDA2_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D10RegisterResource(IntPtr pResource, CUD3D10RegisterFlags Flags);
            /// <summary>
            /// 
            /// </summary>
            /// <param name="pResource"></param>
            /// <returns></returns>
            [Obsolete(CUDA2_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D10UnregisterResource(IntPtr pResource);

            /// <summary>
            /// 
            /// </summary>
            /// <param name="count"></param>
            /// <param name="ppResource"></param>
            /// <returns></returns>
            [Obsolete(CUDA2_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D10MapResources(uint count, ref IntPtr ppResource);
            /// <summary>
            /// 
            /// </summary>
            /// <param name="count"></param>
            /// <param name="ppResource"></param>
            /// <returns></returns>
            [Obsolete(CUDA2_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D10UnmapResources(uint count, ref IntPtr ppResource);
            /// <summary>
            /// 
            /// </summary>
            /// <param name="pResource"></param>
            /// <param name="Flags"></param>
            /// <returns></returns>
            [Obsolete(CUDA2_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D10ResourceSetMapFlags(IntPtr pResource, CUD3D10MapFlags Flags);

            /// <summary>
            /// 
            /// </summary>
            /// <param name="pWidth"></param>
            /// <param name="pHeight"></param>
            /// <param name="pDepth"></param>
            /// <param name="pResource"></param>
            /// <param name="Face"></param>
            /// <param name="Level"></param>
            /// <returns></returns>
            [Obsolete(CUDA2_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D10ResourceGetSurfaceDimensions(ref uint pWidth, ref uint pHeight, ref uint pDepth, IntPtr pResource, uint Face, uint Level);
            /// <summary>
            /// 
            /// </summary>
            /// <param name="pArray"></param>
            /// <param name="pResource"></param>
            /// <param name="Face"></param>
            /// <param name="Level"></param>
            /// <returns></returns>
            [Obsolete(CUDA2_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D10ResourceGetMappedArray(ref CUarray pArray, IntPtr pResource, uint Face, uint Level);
            /// <summary>
            /// 
            /// </summary>
            /// <param name="pDevPtr"></param>
            /// <param name="pResource"></param>
            /// <param name="Face"></param>
            /// <param name="Level"></param>
            /// <returns></returns>
            [Obsolete(CUDA2_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D10ResourceGetMappedPointer(ref CUdeviceptr pDevPtr, IntPtr pResource, uint Face, uint Level);
            /// <summary>
            /// 
            /// </summary>
            /// <param name="pSize"></param>
            /// <param name="pResource"></param>
            /// <param name="Face"></param>
            /// <param name="Level"></param>
            /// <returns></returns>
            [Obsolete(CUDA2_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D10ResourceGetMappedSize(ref uint pSize, IntPtr pResource, uint Face, uint Level);
            /// <summary>
            /// 
            /// </summary>
            /// <param name="pPitch"></param>
            /// <param name="pPitchSlice"></param>
            /// <param name="pResource"></param>
            /// <param name="Face"></param>
            /// <param name="Level"></param>
            /// <returns></returns>
            [Obsolete(CUDA2_OBSOLETE)]
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D10ResourceGetMappedPitch(ref uint pPitch, ref uint pPitchSlice, IntPtr pResource, uint Face, uint Level);
        }
    }

    /// <summary>
    /// Direct3D 11 Interoperability for CUDA 3.x
    /// </summary>
    public sealed class DirectX11
    {
        internal const string CUDA_DRIVER_API_DLL_NAME = "nvcuda";

        /// <summary>
        /// Returns in <c>device</c> the CUDA-compatible device corresponding to the adapter <c>pAdapter</c> obtained from 
        /// <c>IDXGIFactory::EnumAdapters</c>. This call will succeed only if a device on adapter <c>pAdapter</c> is Cuda-compatible.
        /// </summary>
        /// <param name="device">Returned CUDA device corresponding to pszAdapterName</param>
        /// <param name="pAdapter">Adapter (type: IDXGIAdapter)</param>
        /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
        /// <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorUnknown"/>.
        /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
        [DllImport(CUDA_DRIVER_API_DLL_NAME)]
        public static extern CUResult cuD3D11GetDevice(ref CUdevice device, IntPtr pAdapter);
        /// <summary>
        /// Creates a new CUDA context, enables interoperability for that context with the Direct3D device <c>pD3DDevice</c>, and
        /// associates the created CUDA context with the calling thread. The created <see cref="CUcontext"/> will be returned in <c>pCtx</c>.
        /// Direct3D resources from this device may be registered and mapped through the lifetime of this CUDA context.
        /// If <c>pCudaDevice</c> is non-NULL then the <see cref="CUdevice"/> on which this CUDA context was created will be returned in
        /// <c>pCudaDevice</c>.
        /// On success, this call will increase the internal reference count on <c>pD3DDevice</c>. This reference count will be decremented
        /// upon destruction of this context through <see cref="CUDA.DriverAPI.ContextManagement.cuCtxDestroy"/>. This context will cease to function if <c>pD3DDevice</c>
        /// is destroyed or encounters an error.
        /// </summary>
        /// <param name="pCtx">Returned newly created CUDA context</param>
        /// <param name="pCudaDevice">Returned pointer to the device on which the context was created</param>
        /// <param name="Flags">Context creation flags (see <see cref="CUDA.DriverAPI.ContextManagement.cuCtxCreate"/> for details)</param>
        /// <param name="pD3DDevice">Direct3D device to create interoperability context with</param>
        /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
        /// <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorOutOfMemory"/>, <see cref="CUResult.ErrorUnknown"/>.
        /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
        [DllImport(CUDA_DRIVER_API_DLL_NAME)]
        public static extern CUResult cuD3D11CtxCreate(ref CUcontext pCtx, ref CUdevice pCudaDevice, CUCtxFlags Flags, IntPtr pD3DDevice);
        /// <summary>
        /// Registers the Direct3D 11 resource <c>pD3DResource</c> for access by CUDA and returns a CUDA handle to
        /// <c>pD3Dresource</c> in <c>pCudaResource</c>. The handle returned in <c>pCudaResource</c> may be used to map and
        /// unmap this resource until it is unregistered. On success this call will increase the internal reference count on
        /// <c>pD3DResource</c>. This reference count will be decremented when this resource is unregistered through <see cref="CUDA.DriverAPI.GraphicsInterop.cuGraphicsUnregisterResource"/>.<para/>
        /// This call is potentially high-overhead and should not be called every frame in interactive applications.<para/>
        /// The type of pD3DResource must be one of the following:
        /// <list type="table">  
        /// <listheader><term>Type of <c>pD3DResource</c></term><description>Restriction</description></listheader>  
        /// <item><term>ID3D11Buffer</term><description>
        /// May be accessed through a device pointer.
        /// </description></item>  
        /// <item><term>ID3D11Texture1D</term><description>
        /// Individual subresources of the texture may be accessed via arrays.
        /// </description></item>  
        /// <item><term>ID3D11Texture2D</term><description>
        /// Individual subresources of the texture may be accessed via arrays.
        /// </description></item> 
        /// <item><term>ID3D11Texture3D</term><description>
        /// Individual subresources of the texture may be accessed via arrays.
        /// </description></item>  
        /// </list> 
        /// The Flags argument may be used to specify additional parameters at register time. The only valid value for this
        /// parameter is <see cref="CUGraphicsRegisterFlags.None"/>. <para/>
        /// Not all Direct3D resources of the above types may be used for interoperability with CUDA. The following are some
        /// limitations.<param/>
        /// • The primary rendertarget may not be registered with CUDA.<param/>
        /// • Resources allocated as shared may not be registered with CUDA.<param/>
        /// • Textures which are not of a format which is 1, 2, or 4 channels of 8, 16, or 32-bit integer or floating-point data
        /// cannot be shared.<param/>
        /// • Surfaces of depth or stencil formats cannot be shared.<param/>
        /// If Direct3D interoperability is not initialized for this context using <see cref="cuD3D11CtxCreate"/> then
        /// <see cref="CUResult.ErrorInvalidContext"/> is returned. If <c>pD3DResource</c> is of incorrect type or is already registered then
        /// <see cref="CUResult.ErrorInvalidHandle"/> is returned. If <c>pD3DResource</c> cannot be registered then 
        /// <see cref="CUResult.ErrorUnknown"/> is returned. If <c>Flags</c> is not one of the above specified value then <see cref="CUResult.ErrorInvalidValue"/>
        /// is returned.
        /// </summary>
        /// <param name="pCudaResource">Returned graphics resource handle</param>
        /// <param name="pD3DResource">Direct3D resource to register</param>
        /// <param name="Flags">Parameters for resource registration</param>
        /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
        /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidHandle"/>, <see cref="CUResult.ErrorOutOfMemory"/>, <see cref="CUResult.ErrorUnknown"/>.
        /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
        [DllImport(CUDA_DRIVER_API_DLL_NAME)]
        public static extern CUResult cuGraphicsD3D11RegisterResource(ref CUgraphicsResource pCudaResource, IntPtr pD3DResource, CUGraphicsRegisterFlags Flags);
    }
}
