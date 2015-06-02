
from numpy import *
from scipy import *
from scipy.spatial import * 
from matplotlib.pyplot import *

ion()

NodeCount = 1000

X = rand(NodeCount)
Y = rand(NodeCount)

Pts = zeros((NodeCount,2))

Pts[:,0] = X
Pts[:,1] = Y


# Count the number of times each node is referenced 
ElementCount = zeros(NodeCount) 

scatter(X,Y)
show()

Tri = Delaunay(Pts);

triplot(Pts[:,0], Pts[:,1], Tri.simplices.copy())

Arr = Tri.simplices;



print ElementCount	
raw_input('Press any key to continue...')
