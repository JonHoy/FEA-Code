import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.autoinit
import numpy as np

from pycuda.compiler import SourceModule


my_mod = drv.module_from_file("Isoparametric.cubin")

NumElements = 1000;
NodesPerElement = 8

Elements = np.zeros((NodesPerElement,NumElements), dtype=np.int32)

id = 0;
for i in xrange(NodesPerElement):
	for j in xrange(NumElements):
		Elements[i,j] = id
		id = id + 1


print Elements			
			
