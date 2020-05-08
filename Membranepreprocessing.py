import numpy as np
import mrcfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def mrcopen(mrcfilename):
    """Function to open an mrc file and return the data as a numpy array"""
    with mrcfile.open(mrcfilename) as mrc:
       mrcfiledata = np.array(mrc.data)

    return mrcfiledata

def zgradient(array):
    centralslices = array[:, :, (90, 110)]
    centralslices2 = array[:, (90, 110), :]
    # sum those slices for some extra SNR
    centralslicessum = np.sum(centralslices, axis=2)
    centralslicessum2 = np.sum(centralslices2, axis=1)
    #plt.imshow(centralslicessum2)
    #Split the array down the middle.
    split1a, split2a, split3a, split4a = np.split(centralslicessum, 4, axis=1)
    split1b, split2b, split3b, split4b = np.split(centralslicessum2, 4, axis=1)

    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(split1b)
    ax[1].imshow(split2b)
    ax[2].imshow(split3b)
    ax[3].imshow(split4b)
    plt.show()
    #calculate the z values in direction one first
    if np.count_nonzero(split1a) > 0:
        Split1Coords = np.where(split1a > 0)
        Split1max = np.max(Split1Coords[0])
        Split1min = np.min(Split1Coords[0])
    else:
        Split1Coords = np.where(split2a > 0)
        Split2max = np.max(Split1Coords[0])
        Split1min = np.min(Split1Coords[0])

    if np.count_nonzero(split4a) > 0:
        Split2Coords = np.where(split4a > 0)
        Split2max = np.max(Split2Coords[0])
        Split2min = np.min(Split2Coords[0])
    else:
        Split2Coords = np.where(split3a > 0)
        Split2max = np.max(Split2Coords[0])
        Split2min = np.min(Split2Coords[0])

    #calculate the z values of the second direction

    if np.count_nonzero(split1b) > 0:
        Split1Coords = np.where(split1b > 0)
        Split3max = np.max(Split1Coords[0])
        Split3min = np.min(Split1Coords[0])
    else:
        Split1Coords = np.where(split2b > 0)
        Split3max = np.max(Split1Coords[0])
        Split3min = np.min(Split1Coords[0])

    if np.count_nonzero(split4b) > 0:
        Split4Coords = np.where(split4b > 0)
        Split2max = np.max(Split2Coords[0])
        Split4min = np.min(Split2Coords[0])
    else:
        Split2Coords = np.where(split3b > 0)
        Split2max = np.max(Split2Coords[0])
        Split4min = np.min(Split2Coords[0])


    Zgradient = Split1min - Split2min
    Zgradient2 = Split3min - Split4min
    return Zgradient, Zgradient2


def Zvector(array):
    #get the central slices and sum for some extra SNR
    centralslices = array[:, :, (90, 110)]
    centralslices2 = array[:, (90, 110), :]
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
    #plt.show()
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
    #plt.show()

    return YZ, XZ

#open the membrane mrc file
membrane = mrcopen('job907membrane2.mrc')

#filter according to chimera so that there is only membrane density in the volume
membrane[membrane < 0.00785] = 0
#plt.imshow(membrane[:,:,100])

YZ, XZ = Zvector(membrane)

VectorAlongX = (200, 0, XZ)
VectorAlongY = (0, 200, YZ)

Cross = np.cross(VectorAlongX, VectorAlongY)
Cross = Cross / np.linalg.norm(Cross)
point = np.array([100,100,100])
xx, yy = np.meshgrid(range(200), range(200))
d = -point.dot(Cross)
zz = (-Cross[0] * xx - Cross[1] * yy -d) * 1. / Cross[2]
threedimensionplot = np.where(membrane > 0)
threedimensionplotx = threedimensionplot[2][1::75]
threedimensionploty = threedimensionplot[1][1::75]
threedimensionplotz = threedimensionplot[0][1::75]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(threedimensionplotx, threedimensionploty, threedimensionplotz)
ax.plot_surface(xx,yy, zz, color='y')
ax.set_xlim(0,200)
ax.set_ylim(0,200)
ax.set_zlim(0,200)
plt.show()

