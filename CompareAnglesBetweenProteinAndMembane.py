
#Imports

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from tqdm import tqdm
#from scipy.spatial.transform import Rotation as R

### -------------- USER INPUTS -------------- ###

# Provide name of star file
ProteinStar = 'Evas_data/HA.star'
MembraneStar = 'Evas_data/membrane.star'

# +/- central particle to search, this may be useful to mitigate effect of missing wedge on results.
Zrange = 1000

# Output star file of particles between two angles (True or False followed by the angles to search between):

CreateStarFile = 'false'

FirstAngle = 0
SecondAngle = 2

# Set which graphs are required by adding True of False:

HISTOGRAM = 'True'
HISTOGRAMNAME = 'Evas_data/histrecent.png'

DISTRIBUTIONPLOT = 'False'
DISTRIBUTIONPLOTNAME = 'Dist15042020.png'

VECTORSPLOT = 'False'
VECTORSPLOTNAME = 'Vect15042020'

#Write out list of angles:
AngleListOutput = 'False'

ProVector = (0, 0, 1)
MemVector = (0, 0, 1)

# -------- Functions -------- #


'''
Functions For dealing with Star files from relion 3.0 or earlier.
'''

def OpenStarFile(StarFileName):
    titles = []
    data = open(StarFileName)
    datalines = data.readlines()
    counter = 0
    for line in range(len(datalines) - 1):
        if '_rln' in datalines[line]:
            var = datalines[line]
            var = var.strip(' \n')
            var = var.strip('_rln')
            var = var.replace(' #', '')
            var = var.strip('\r')
            var = ''.join(i for i in var if not i.isdigit())  # These lines just remove the stuff around the headings
            var = var.strip(' ')
            titles.append(var)
            counter = line

    dataframe = pd.read_csv(StarFileName, delim_whitespace=True, skiprows=counter+1, header=None)
    dataframe.columns = titles
    return dataframe

def OutputStarFile(dataframe, NameOfOutput):

    OutputStar = ['data_', 'loop_']
    header = dataframe.columns.values

    for title in range(len(header)):
        currenthead = header[title]
        prehead = '_rln'
        posthead = ' #{}'.format(title + 1)
        NewTitle = '{}{}{}'.format(prehead, currenthead, posthead)
        OutputStar.append(NewTitle)
    data = dataframe.values

    for line in range(len(data)):
        information = data[line, :]
        OutputLine = []
        for bit in range(len(information)):
            OutputLine.append('{}   '.format(information[bit]))
        OutputLine = ''.join(OutputLine)
        OutputStar.append(OutputLine)

    with open(NameOfOutput, "w") as myfile:
        for item in range(len(OutputStar)):
            output = OutputStar[item]
            myfile.write('{}\n'.format(output))

'''
Functions for carrying out rotations of 3D vectors
'''

def RotationMatrix(axis, rotang):
    """
    This uses Euler-Rodrigues formula.
    """
    #Input taken in degrees, here we change it to radians
    theta = rotang * 0.0174532925
    axis = np.asarray(axis)
    #Ensure axis is a unit vector
    axis = axis/math.sqrt(np.dot(axis, axis))
    #calclating a, b, c and d according to euler-rodrigues forumla requirments
    a = math.cos(theta/2)
    b, c, d = axis*math.sin(theta/2)
    a2, b2, c2, d2 = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    #Return the rotation matrix
    return np.array([[a2+b2-c2-d2, 2*(bc-ad), 2*(bd+ac)],
                     [2*(bc+ad), a2+c2-b2-d2, 2*(cd-ab)],
                     [2*(bd-ac), 2*(cd+ab), a2+d2-b2-c2]])

def BelnapMethod(Rot, Tilt, Psi):

    Rotradian = math.radians(Rot)
    Tiltradian = math.radians(Tilt)
    Psiradian = math.radians(Psi)

    SinRot = math.sin(Rotradian)
    SinTilt = math.sin(Tiltradian)
    SinPsi = math.sin(Psiradian)

    CosRot = math.cos(Rotradian)
    CosTilt = math.cos(Tiltradian)
    CosPsi = math.cos(Psiradian)

    return np.array([[(CosPsi*CosTilt*CosRot) - (SinPsi*SinRot), (CosPsi*CosTilt*SinRot) + (SinPsi*CosRot), (-CosPsi*SinTilt)],
                     [(-SinPsi*CosTilt*CosRot) - (CosPsi*SinRot), (-SinPsi*CosTilt*SinRot) + (CosPsi*CosRot), (SinPsi*SinTilt)],
                     [(SinTilt*CosRot), (SinTilt*SinRot), (CosTilt)]])



def ApplyRotationMatrix(vector, rotationmatrix):
    """
    This function take the output from the RotationMatrix function and
    uses that to apply the rotation to an input vector
    """
    a1 = (vector[0] * rotationmatrix[0, 0]) + (vector[1] * rotationmatrix[0, 1]) + (vector[2] * rotationmatrix[0, 2])
    b1 = (vector[0] * rotationmatrix[1, 0]) + (vector[1] * rotationmatrix[1, 1]) + (vector[2] * rotationmatrix[1, 2])
    c1 = (vector[0] * rotationmatrix[2, 0]) + (vector[1] * rotationmatrix[2, 1]) + (vector[2] * rotationmatrix[2, 2])

    return np.array((a1, b1, c1))

def RotationMethod(Rot, Tilt, Psi, Vector):
    """
    This method takes the Euler angles from relion and the vector that describes the protein or normal
    to the membrane and uses the two above functions to return the vector to the orientation the object
    would have been before alignment.
    """

    #NOTE:

    #All axes have to be calculated first.
    RotateMatrixRot = RotationMatrix((0, 0, 1), Rot)
    YPrime = ApplyRotationMatrix((0, 1, 0), RotateMatrixRot)
    RotateMatrixTilt = RotationMatrix(YPrime, Tilt)
    ZPrimePrime = ApplyRotationMatrix((0, 0, 1), RotateMatrixTilt)
    RotateMatrixPsi = RotationMatrix(ZPrimePrime, Psi)

    #Rotations then applied to the input vector.

    RotatedZ = ApplyRotationMatrix(Vector, RotateMatrixRot)
    RotatedZY = ApplyRotationMatrix(RotatedZ, RotateMatrixTilt)
    RotatedZ2 = ApplyRotationMatrix(RotatedZY, RotateMatrixPsi)

    return RotatedZ2



'''
Functions for Calculating the angles of 3D vectors relative to one another
'''

def CalculateAngleBetweenVector(vector, vector2):
    """
    Does what it says on the tin, outputs an angle in degrees between two input vectors.
    """
    dp = np.dot(vector, vector2)

    maga = math.sqrt((vector[0] ** 2) + (vector[1] ** 2) + (vector[2] ** 2))
    magb = math.sqrt((vector2[0] ** 2) + (vector2[1] ** 2) + (vector2[2] ** 2))
    magc = maga * magb

    dpmag = dp / magc

    #These if statements deal with rounding errors of floating point operations
    if dpmag > 1:
        error = dpmag - 1
        print('error = {}, do not worry if this number is very small'.format(error))
        dpmag = 1
    elif dpmag < -1:
        error = 1 + dpmag
        print('error = {}, do not worry if this number is very small'.format(error))
        dpmag = -1


    angleindeg = ((math.acos(dpmag)) * 180) / math.pi

    return angleindeg


def CalculateAngleAroundZ(Vector):
    X,Y,Z = Vector[0], Vector[1], Vector[2]
    AngleAroundZ = math.atan2(Y, X)

    return AngleAroundZ


def AngleOffset(Vector1, Vector2):
    # Takes the two input vectors, calculates the offset and the RM required to rotate one onto the other.
    # ensure things are unit vectors
    Vector1 = Vector1 / np.linalg.norm(Vector1)
    Vector2 = Vector2 / np.linalg.norm(Vector2)
    # calculate the offset and axis of rotation to take vector to the z axis
    angle = CalculateAngleBetweenVector(Vector1, Vector2)
    cross = np.cross(Vector1, Vector2)
    RM = RotationMatrix(cross, angle)

    return RM

#------------- MAIN CODE STARTS -------------#

print('Pre-processing...')

# Open both star files
file1 = OpenStarFile(ProteinStar)
file2 = OpenStarFile(MembraneStar)

#Create the rotation matrix that defines the rotation between the two input vectors. This allows normalisation of all the vectors.
#This is added as an additional rotation when carrying out rotations.
OffsetRotationMatrix = AngleOffset(ProVector, MemVector)

# Get the values needed from the star files
# More info taken from file 1 for calculating particle location in z, this is used later on
file1_array_df = file1[['ImageName', 'AngleRot', 'AngleTilt', 'AnglePsi', 'CoordinateZ', 'MicrographName']]
file2_array = (file2[['ImageName', 'AngleRot', 'AngleTilt', 'AnglePsi']].values)

# Generating a list of names of each tomogram. This is used when calculating the central z slice in a tomogram.
file1_Names = file1_array_df.MicrographName.unique()

# Transfer values over to numpy array.
file1_array = file1_array_df.values

#Create an empty list to populate with all the angles.
anglelist = []
Tilts = []
#Create empty list to populate with angles of proteins within angle ranges
SpecificAngleList = []

#Set up figures to be appended as code runs
fig = plt.figure('3D plot of vectors +/- 1000 virosomes')
ax = Axes3D(fig)
ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(-1, 1)

fig2 = plt.figure('Angular distribution plot Virosomes +/- 1')
ax2 = plt.subplot(111, projection='polar')
ax2.set_title('Angular distribution plot Virus +/- 100 Z')

print('Main processing...')
#iterate through each array, tqdm provides a loading bar.
for n in tqdm(range(0, (file1_array.shape[0] - 1))):
    for n2 in range(0, (file2_array.shape[0] - 1)):
        # Iterating through all the lines until a protein and membrane with the same name are found
        if file1_array[n, 0].split('/')[3] == (file2_array[n2, 0].split('/'))[3]:
            # pull out the tomogram name
            filename = (file1_array[n, 5])
            # Generate a temp array with all coordinates of particles in that tomogram
            file1_array_currenttomo = file1_array_df[file1_array_df.MicrographName.isin([filename])]
            #  Pull out all Z values of the coordinates in that array
            zvalues = pd.to_numeric(file1_array_currenttomo['CoordinateZ'])
            # Find the central z coordinate between all particles and assume this to be the central slice.
            sum = zvalues.sum()
            CentralSlice = sum / file1_array_currenttomo.shape[0]
            CentralSlice = CentralSlice.astype(int)
            # Define ranges in z to allow to the next stage of processing
            RangeTop = CentralSlice + Zrange
            RangeBot = CentralSlice - Zrange
            # Assuming everything is in order the angle is now worked out
            if file1_array[n, 4] < RangeTop and file1_array[n, 4] > RangeBot:

                # Protein angles
                Protein_Rot = file1_array[n, 1] * -1
                Protein_Tilt = file1_array[n, 2] * -1
                Protein_Psi = file1_array[n, 3] * -1
                # Membrane angles
                Mem_Rot = file2_array[n2, 1] * -1
                Mem_Tilt = file2_array[n2, 2] * -1
                Mem_Psi = file2_array[n2, 3] * -1

                # Because membrane and protein have been aligned with symmetry applied they both line up on the z axis
                # This normalises the normal to the membrane to 0,0,1 and the HA to a vector around this.
                Rot = Protein_Rot - Mem_Rot
                Tilt = Protein_Tilt - Mem_Tilt
                Tilts.append(Tilt)
                Psi = Protein_Psi - Mem_Psi
                # Apply the rotation method function
                ProteinRotated = RotationMethod(Psi, Tilt, Rot, (0,0,1))
                #Apple the offset if needed
                if ProVector != MemVector:
                    ProteinRotated = ApplyRotationMatrix(ProteinRotated, OffsetRotationMatrix)

                # Calculate Unit vectors
                ProteinRotatedUV = ProteinRotated / np.linalg.norm(ProteinRotated)

                # Angle calculation
                angle = CalculateAngleBetweenVector(ProteinRotatedUV, (0,0,1))

                # Append that list of angles I made earlier
                anglelist.append(angle)

                # THIS NEEDS FIXING AFTER EULER DRAMA
                # Here I am generating a vector to plot the angle of the HA relative to the membrane on a 3D quiver graph
                VectorForPlotting = np.array([0, 0, 0, ProteinRotatedUV[0], ProteinRotatedUV[1], ProteinRotatedUV[2]])
                ax.quiver(VectorForPlotting[0], VectorForPlotting[1], VectorForPlotting[2], VectorForPlotting[3],
                          VectorForPlotting[4], VectorForPlotting[5], pivot='tail', linewidths=1)

                #THIS NEEDS FIXING AFTER EULER DRAMA
                # I also wanted to plot the angular distribution so here I calculate the angle around the Z axis
                AngleAroundZ = CalculateAngleAroundZ(ProteinRotatedUV)
                AngleAroundZ = AngleAroundZ * (180 / math.pi)
                # THIS NEEDS FIXING AFTER EULER DRAMA
                # Here I add to the angular distribution graph THIS NEEDS FIXING AFTER EULER DRAMA
                c = ax2.scatter((Protein_Rot + Protein_Psi), angle, s=1.5, cmap='nipy_spectral', alpha=0.75)

                # Next I can output a text file of particle names at a specific angle
                if CreateStarFile == 'True':
                    if FirstAngle < angle < SecondAngle:
                        SpecificAngleList.append(file1_array[n, 0])




print('Writing output files...')
# make histogram
fig3 = plt.figure()
ax3 = plt.subplot(111)
plt.hist(anglelist, bins=100, density=True)
# Save all the figures!

if HISTOGRAM == 'True' or 'true':
    fig3.savefig(HISTOGRAMNAME)

if DISTRIBUTIONPLOT == 'True' or 'true':
   fig2.savefig(DISTRIBUTIONPLOTNAME, dpi=300)

if VECTORSPLOT == 'True' or 'true':
    fig.savefig(VECTORSPLOTNAME)

# Other outputs:
if CreateStarFile == 'True' or 'true':
    Outputdf = file1[file1.ImageName.isin(SpecificAngleList)]
    OutputStarFile(Outputdf, 'StarFileOfProteinsBetweenAngles{}and{}.star'.format(FirstAngle, SecondAngle))

if AngleListOutput == 'True' or 'true':
    with open('Angles.txt', 'w') as f:
        for item in anglelist:
            f.write("%s\n" % item)