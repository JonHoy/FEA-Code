using System;
using System.Collections.Generic;
using System.Text;
using System.Windows.Threading;

namespace ManagedCuda.Dispatched
{
    /// <summary>
    /// Dispatched CUDA base class
    /// </summary>
    [Obsolete("CUDA 4.0 API calls are now thread safe. Dispatched... wrapper classes are of no use now.")]
    public abstract class DispatchedCudaBaseClass
    {
        /// <summary>
        /// Dispatcher managing all methods calls of the CUDA thread
        /// </summary>
        protected Dispatcher _dispatcher;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dispatcher"></param>
        public DispatchedCudaBaseClass(Dispatcher dispatcher)
        {
            _dispatcher = dispatcher;
        }

        /// <summary>
        /// dispatcher member must be set manually
        /// </summary>
        public DispatchedCudaBaseClass()
        {

        }

        /// <summary>
        /// 
        /// </summary>
        public Dispatcher Dispatcher
        {
            get { return _dispatcher; }
            set { _dispatcher = value; }
        }
    }
}
