import numpy as np
import mrcfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

###-------Variables--------####

#Path to the mrc file of the membrane to calculate the normal for
MembranePath = 'emd_25833.map'

#Contour level for display of the membrane
MembraneContour = 0.03111

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

        #get the central slices and sum for some extra SNR
        centralslicesX = FilteredArray[:, :, (int((FilteredArray.shape[2] / 2)) - 20):(int((FilteredArray.shape[2] / 2)) + 20)]
        centralslicesY = FilteredArray[:, (int((FilteredArray.shape[2] / 2)) - 20):(int((FilteredArray.shape[2] / 2)) + 20), :]
        centralslicessumX = np.sum(centralslicesX, axis=2)
        centralslicessumY = np.sum(centralslicesY, axis=1)


        #find locations where the array is greater than 0 to create scatter plot
        Ydirection = np.where(centralslicessumY > 0)
        Xdirection = np.where(centralslicessumX > 0)

        #fit a line to the locations above 0
        Ym, Yc = np.polyfit(Ydirection[1], Ydirection[0], 1)
        Xm, Xc = np.polyfit(Xdirection[1], Xdirection[0], 1)

        #use equation for a line to calculate the z values
        Yzmin = Ym * (0) + Yc
        Yzmax = Ym * (200) + Yc
        Xzmin = Xm * 0 + Xc
        Xzmax = Xm * 200 + Xc

        fig, ax = plt.subplots(1, 2)
        ax[0].scatter(Ydirection[1], Ydirection[0])
        ax[0].plot(Ydirection[1], Ym * Ydirection[1] + Yc, color='y')
        ax[1].scatter(Xdirection[1], Xdirection[0])
        ax[1].plot(Xdirection[1], Xm * Xdirection[1] + Xc, color='y')
        plt.setp(ax, xlim=(0, FilteredArray.shape[2]), ylim=(0, FilteredArray.shape[1]))

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

        return YZ, XZ, Cross

#open the membrane mrc file
Membrane = MembraneProcessor(MembranePath)
YZ, XZ, Cross = Membrane.Zvector(ContourLevel=MembraneContour)
print('Use this vector as the membrane vector for comparing angles between protein and membrane: {}'.format(Cross))

#Everything from here on out could techniqually  be a function, it plots everything in a three dimensional
#graph so you can view the structure, the membrane and hte vector described in the print above.
VectorAlongX = (200, 0, XZ)
VectorAlongY = (0, 200, YZ)

point = np.array([Membrane.MembraneArray.shape[0]/2,Membrane.MembraneArray.shape[1]/2,Membrane.MembraneArray.shape[2]/2])
xx, yy = np.meshgrid(range(Membrane.MembraneArray.shape[1]), range(Membrane.MembraneArray.shape[1]))
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
ax.quiver((Membrane.MembraneArray.shape[0] / 2),(Membrane.MembraneArray.shape[1] / 2),
          (Membrane.MembraneArray.shape[2] / 2), Cross[0], Cross[1], Cross[2],
          length=(Membrane.MembraneArray.shape[2] / 3), normalize=1, color='red')
plt.show()

