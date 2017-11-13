import numpy as np
import matplotlib.pyplot as plt
import colorsys
import sys

K = 3   # number of centroids to compute
numClusters = 3 # actual number of clusters to generate
ptsPerCluster = 40  # number of points per actual cluster
xCenterBounds = (-2, 2)    # lower and upper limits within which to place actual cluster centers

# Randomly place cluster centers within the span of xCenterBounds.
centers = np.random.random_sample((numClusters,))
centers = centers * (xCenterBounds[1] - xCenterBounds[0]) + xCenterBounds[0]

# Initialize array of data points.
points = np.zeros((numClusters * ptsPerCluster,))

# Normally distribute ptsPerCluster points around each center.
stDev = 0.15
for i in range(numClusters):
    points[i*ptsPerCluster:(i+1)*ptsPerCluster] = (
        stDev * np.random.randn(ptsPerCluster) + centers[i])

# Randomly select K points as the initial centroid locations.
centroids = np.zeros((K,))
indices = []
while len(indices) < K:
    index = np.random.randint(0, numClusters * ptsPerCluster)
    if not index in indices:
        indices.append(index)
centroids = points[indices]

# Assign each point to its nearest centroid. Store this in classifications,
# where each element will be an int from 0 to K-1.
classifications = np.zeros((points.shape[0],), dtype=np.int)
def assignPointsToCentroids():
    for i in range(points.shape[0]):
        smallestDistance = 0
        for k in range(K):
            distance = abs(points[i] - centroids[k])
            if k == 0:
                smallestDistance = distance
                classifications[i] = k
            elif distance < smallestDistance:
                smallestDistance = distance
                classifications[i] = k

assignPointsToCentroids()

# Define a function to recalculate the centroid of a cluster.
def recalcCentroids():
    for k in range(K):
        if sum(classifications == k) > 0:
            centroids[k] = sum(points[classifications == k]) / sum(classifications == k)

# Generate a unique color for each of the K clusters using the HSV color scheme.
# Simultaneously, initialize matplotlib line objects for each centroid and cluster.
hues = np.linspace(0, 1, K+1)[:-1]

fig, ax = plt.subplots()
clusterPointsList = []
centroidPointsList = []
for k in range(K):
    clusterColor = tuple(colorsys.hsv_to_rgb(hues[k], 0.8, 0.8))

    clusterLineObj, = ax.plot([], [], ls='None', marker='x', color=clusterColor)
    clusterPointsList.append(clusterLineObj)

    centroidLineObj, = ax.plot([], [], ls='None', marker='o', 
        markeredgecolor='k', color=clusterColor)
    centroidPointsList.append(centroidLineObj)
iterText = ax.annotate('', xy=(0.01, 0.01), xycoords='axes fraction')

# Define a function to update the plot.
def updatePlot(iteration):
    for k in range(K):
        xDataNew = points[classifications == k]
        clusterPointsList[k].set_data(xDataNew, np.zeros((len(xDataNew),)))
        centroidPointsList[k].set_data(centroids[k], 0)
    iterText.set_text('i = {:d}'.format(iteration))
    plt.savefig('./images/{:d}.png'.format(iteration))
    plt.pause(0.5)

dataRange = np.amax(points) - np.amin(points)
ax.set_xlim(np.amin(points) - 0.05 * dataRange, np.amax(points) + 0.05 * dataRange)
ax.set_ylim(-1, 1)
iteration = 0
updatePlot(iteration)
plt.ion()
plt.show()

# Execute and animate the algorithm with a while loop. Note that this is not the
# best way to animate a matplotlib plot--the matplotlib animation module should be
# used instead, but we will use a while loop here for simplicity.
lastCentroids = centroids + 1
while not np.array_equal(centroids, lastCentroids):
    lastCentroids = np.copy(centroids)
    recalcCentroids()
    assignPointsToCentroids()
    iteration += 1
    updatePlot(iteration)

pythonMajorVersion = sys.version_info[0]
if pythonMajorVersion < 3:
    raw_input("Press Enter to continue.")
else:
    input("Press Enter to continue.")
