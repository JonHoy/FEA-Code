using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Text;

namespace ManagedCuda.NVRTC
{	
	/// <summary>
	/// CUDA Online Compiler API call result code.
	/// </summary>
	public enum nvrtcResult
	{
		Success = 0,
		ErrorOutOfMemory = 1,
		ErrorProgramCreationFailure = 2,
		ErrorInvalidInput = 3,
		ErrorInvalidProgram = 4,
		ErrorInvalidOption = 5,
		ErrorCompilation = 6,
		ErrorBuiltinOperationFailure = 7
	}

	/// <summary>
	/// the unit of compilation, and an opaque handle for a program.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct nvrtcProgram
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Pointer;
	}
}
