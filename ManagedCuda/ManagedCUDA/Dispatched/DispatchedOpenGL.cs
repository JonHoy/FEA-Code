using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;
using CUDA.BasicTypes;
using CUDA.VectorTypes;

namespace CUDA.Dispatched
{
    /// <summary>
    /// OpenGL Interoperability
    /// </summary>
    public sealed class OpenGL
    {        
        internal const string CUDA_DRIVER_API_DLL_NAME = "nvcuda";
        internal const string CUDA2_OBSOLETE = "CUDA 2.x compatibility API. These functions are deprecated, please use the CUDA3 ones.";
       
        /// <summary>
        /// OpenGL Interoperability for CUDA 3.x
        /// </summary>
        public sealed class CUDA3
        {
            /// <summary>
            /// Creates a new CUDA context, initializes OpenGL interoperability, and associates the CUDA context with the calling
            /// thread. It must be called before performing any other OpenGL interoperability operations. It may fail if the needed
            /// OpenGL driver facilities are not available. For usage of the Flags parameter, see <see cref="CUDA.DriverAPI.ContextManagement.cuCtxCreate"/>.
            /// </summary>
            /// <param name="pCtx">Returned CUDA context</param>
            /// <param name="Flags">Options for CUDA context creation</param>
            /// <param name="device">Device on which to create the context</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorOutOfMemory"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuGLCtxCreate(ref CUcontext pCtx, CUCtxFlags Flags, CUdevice device);

            /// <summary>
            /// Registers the buffer object specified by buffer for access by CUDA. A handle to the registered object is returned as
            /// <c>pCudaResource</c>. The map flags <c>Flags</c> specify the intended usage.
            /// </summary>
            /// <param name="pCudaResource">Pointer to the returned object handle</param>
            /// <param name="buffer">name of buffer object to be registered</param>
            /// <param name="Flags">Map flags</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorInvalidHandle"/>, 
            /// <see cref="CUResult.ErrorAlreadyMapped"/>, <see cref="CUResult.ErrorInvalidContext"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuGraphicsGLRegisterBuffer(ref CUgraphicsResource pCudaResource, uint buffer, CUGraphicsRegisterFlags Flags);

            /// <summary>
            /// Registers the texture or renderbuffer object specified by <c>image</c> for access by CUDA. <c>target</c> must match the type
            /// of the object. A handle to the registered object is returned as <c>pCudaResource</c>. The map flags Flags specify the
            /// intended usage. <para/>
            /// The following image classes are currently disallowed: <para/>
            /// • Textures with borders <para/>
            /// • Multisampled renderbuffers
            /// </summary>
            /// <param name="pCudaResource">Pointer to the returned object handle</param>
            /// <param name="image">name of texture or renderbuffer object to be registered</param>
            /// <param name="target">Identifies the type of object specified by <c>image</c>, and must be one of <c>GL_TEXTURE_2D</c>,
            /// <c>GL_TEXTURE_RECTANGLE</c>, <c>GL_TEXTURE_CUBE_MAP</c>, 
            /// <c>GL_TEXTURE_3D</c>, <c>GL_TEXTURE_2D_ARRAY</c>, or <c>GL_RENDERBUFFER</c>.</param>
            /// <param name="Flags">Map flags</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorInvalidHandle"/>, 
            /// <see cref="CUResult.ErrorAlreadyMapped"/>, <see cref="CUResult.ErrorInvalidContext"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuGraphicsGLRegisterImage(ref CUgraphicsResource pCudaResource, uint image, ulong target, CUGraphicsRegisterFlags Flags);

            /// <summary>
            /// Returns in <c>pDevice</c> the CUDA device associated with a <c>hGpu</c>, if applicable.
            /// </summary>
            /// <param name="pDevice">Device associated with hGpu</param>
            /// <param name="hGpu">Handle to a GPU, as queried via <c>WGL_NV_gpu_affinity()</c></param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuWGLGetDevice(ref CUdevice pDevice, IntPtr hGpu);
        }
        
        /// <summary>
        /// OpenGL Interoperability for CUDA 2.x
        /// </summary>
        [Obsolete(CUDA2_OBSOLETE)]
        public sealed class CUDA2
        {            
            /// <summary>
            /// Flags to map or unmap a resource
            /// </summary>
            public enum CUGLMapResourceFlags
            {
                /// <summary>
                /// 
                /// </summary>
                NONE          = 0x00,
                /// <summary>
                /// 
                /// </summary>
                READ_ONLY     = 0x01,
                /// <summary>
                /// 
                /// </summary>
                WRITE_DISCARD = 0x02,    
            }

            /// <summary>
            /// 
            /// </summary>
            /// <returns></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuGLInit();
            /// <summary>
            /// 
            /// </summary>
            /// <param name="buffer"></param>
            /// <returns></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuGLRegisterBufferObject(uint buffer);
            /// <summary>
            /// 
            /// </summary>
            /// <param name="dptr"></param>
            /// <param name="size"></param>
            /// <param name="buffer"></param>
            /// <returns></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuGLMapBufferObject(ref CUdeviceptr dptr, ref uint size, uint buffer);
            /// <summary>
            /// 
            /// </summary>
            /// <param name="buffer"></param>
            /// <returns></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuGLUnmapBufferObject(uint buffer);
            /// <summary>
            /// 
            /// </summary>
            /// <param name="buffer"></param>
            /// <returns></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuGLUnregisterBufferObject(uint buffer);

            /// <summary>
            /// 
            /// </summary>
            /// <param name="buffer"></param>
            /// <param name="Flags"></param>
            /// <returns></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuGLSetBufferObjectMapFlags(uint buffer, uint Flags);
            /// <summary>
            /// 
            /// </summary>
            /// <param name="dptr"></param>
            /// <param name="size"></param>
            /// <param name="buffer"></param>
            /// <param name="hStream"></param>
            /// <returns></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuGLMapBufferObjectAsync(ref CUdeviceptr dptr, ref uint size, uint buffer, CUstream hStream);
            /// <summary>
            /// 
            /// </summary>
            /// <param name="buffer"></param>
            /// <param name="hStream"></param>
            /// <returns></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuGLUnmapBufferObjectAsync(uint buffer, CUstream hStream);

        }
    }
}
