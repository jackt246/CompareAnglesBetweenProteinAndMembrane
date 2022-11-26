import numpy as np
import mrcfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

###-------Variables--------####

#Path to the mrc file of the membrane to calculate the normal for
MembranePath = 'emd_26210.map'

#Contour level for display of the membrane
MembraneContour = 130

###############################

class MembraneProcessor():
    #When instantuating the class provide the membrane location
    def __init__(self, MembraneLoc):
        self.MembraneLoc = MembraneLoc
        self.MembraneArray = self.mrcopen()

    def mrcopen(self):
        """Function to open an mrc file and return the data as a numpy array"""
        with mrcfile.open(self.MembraneLoc) as mrc:
            MembraneArray = np.array(mrc.data)
        return MembraneArray

    def FilterArray(self, ContourLevel):
        # filter according to chimera so that there is only membrane density in the volume
        ForFiltering = self.MembraneArray
        ForFiltering[ForFiltering < ContourLevel] = 0
        return ForFiltering

    def Zvector(self, ContourLevel):
        #Filtered Array
        FilteredArray = self.FilterArray(ContourLevel=ContourLevel)
        print(FilteredArray.shape)

        #get the central slices and sum for some extra SNR
        centralslices = FilteredArray[:, :, (FilteredArray.shape[2] - 20):(FilteredArray.shape[2] + 20)]
        centralslices2 = FilteredArray[:, (FilteredArray.shape[2] - 20):(FilteredArray.shape[2] + 20), :]
        centralslicessum = np.sum(centralslices, axis=2)
        centralslicessum2 = np.sum(centralslices2, axis=1)

        #find locations where the array is greater than 0 to create scatter plot
        Ydirection = np.where(centralslicessum > 0)
        Xdirection = np.where(centralslicessum2 > 0)

        #fit a line to the locations above 0
        Ym, Yc = np.polyfit(Ydirection[1], Ydirection[0], 1)
        Xm, Xc = np.polyfit(Xdirection[1], Xdirection[0], 1)

        #plot figure if needed
        fig, ax = plt.subplots(1, 2)
        ax[0].scatter(Ydirection[1], Ydirection[0])
        ax[0].plot(Ydirection[1], Ym * Ydirection[1] + Yc, color='y')
        ax[1].scatter(Xdirection[1], Xdirection[0])
        ax[1].plot(Xdirection[1], Xm * Xdirection[1] + Xc, color='y')
        plt.setp(ax, xlim=(0,200), ylim=(0,200))
        plt.show()
        #use equation for a line to calculate the z values
        Yzmin = Ym * (0) + Yc
        Yzmax = Ym * (200) + Yc
        Xzmin = Xm * 0 + Xc
        Xzmax = Xm * 200 + Xc

        #Calculate the z direction for the vectors
        YZ = Yzmax - Yzmin
        XZ = Xzmax - Xzmin

        #create the full vectors and unit vectors in each case
        VectorAlongX = (200, 0, XZ)
        VectorAlongX = VectorAlongX / np.linalg.norm(VectorAlongX)
        VectorAlongY = (0, 200, YZ)
        VectorAlongY = VectorAlongY / np.linalg.norm(VectorAlongY)

        #cross product
        Cross = np.cross(VectorAlongX, VectorAlongY)
        Cross = Cross / np.linalg.norm(Cross)

        #plot the cross product as a vector on the graphs earlier generated
        ax[0].quiver(100, 113.631, Cross[1], Cross[2], scale = 5)
        ax[0].quiver(100, 113.631, Cross[2], Cross[1] * -1, scale=5)
        ax[1].quiver(100, 120, Cross[0], Cross[2], scale=5)
        ax[1].quiver(100, 120, Cross[2], Cross[0]* -1, scale=5)
        plt.show()

        return YZ, XZ

#open the membrane mrc file
Membrane = MembraneProcessor(MembranePath)
YZ, XZ = Membrane.Zvector(ContourLevel=MembraneContour)

VectorAlongX = (200, 0, XZ)
VectorAlongY = (0, 200, YZ)

Cross = np.cross(VectorAlongX, VectorAlongY)
Cross = Cross / np.linalg.norm(Cross)
print(Cross)
point = np.array([100,100,100])
xx, yy = np.meshgrid(range(200), range(200))
d = -point.dot(Cross)
zz = (-Cross[0] * xx - Cross[1] * yy -d) * 1. / Cross[2]
threedimensionplot = np.where(Membrane.FilterArray(ContourLevel=MembraneContour) > 0)
threedimensionplotx = threedimensionplot[2][1::25]
threedimensionploty = threedimensionplot[1][1::25]
threedimensionplotz = threedimensionplot[0][1::25]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx,yy, zz, color='y')
ax.scatter(threedimensionplotx, threedimensionploty, threedimensionplotz)

ax.set_xlim(0,200)
ax.set_ylim(0,200)
ax.set_zlim(0,200)
plt.show()

