from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import os


cd = '/home/jonathan/Desktop/FEA-Code/FEA/Unit_Tests/bin/Debug/'
Contents = os.listdir(cd)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

counter = 0;
Filename = "NURB" + str(counter) + ".dat"
while True:
	
	Filename = "NURB" + str(counter) + ".dat"
	if Filename not in Contents:
		break
	File = open(cd + Filename)
	DataStr = File.read()
	Raw = np.fromstring(DataStr)
	counter = counter + 1
	X = np.zeros(len(Raw) / 3)
	Y = np.zeros(len(Raw) / 3)
	Z = np.zeros(len(Raw) / 3)		
	for i in xrange(len(X)):
		X[i] = Raw[3*i]
		Y[i] = Raw[3*i + 1]
		Z[i] = Raw[3*i + 2] 
	ax.plot_wireframe(X, Y, Z)



plt.show()
