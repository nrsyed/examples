from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import colorsys
import sys
from datetime import datetime
from KMeansND import *

K = 12   # Number of clusters (centroids) to compute
numClusters = 12     # Number of clusters to generate
ptsPerCluster = 100
varianceCoeff = 0.01    # This controls the spread of clustered points

xCenterBounds = (-4, 4)
yCenterBounds = (-4, 4)

covariance = np.array([[varianceCoeff * (xCenterBounds[1] - xCenterBounds[0]), 0],
                       [0, varianceCoeff * (yCenterBounds[1] - yCenterBounds[0])]])

def generateClusters():
    centers = np.random.random_sample((numClusters, 2))
    centers[:,0] = (
        centers[:,0] * (xCenterBounds[1] - xCenterBounds[0]) + xCenterBounds[0])
    centers[:,1] = (
        centers[:,1] * (yCenterBounds[1] - yCenterBounds[0]) + yCenterBounds[0])

    points = np.zeros((numClusters * ptsPerCluster, 2))
    for i in range(numClusters):
        points[i*ptsPerCluster : (i+1)*ptsPerCluster,:] = (
            np.random.multivariate_normal(centers[i,:], covariance, ptsPerCluster))
    return points

def initializeCentroids(K, points):
    '''Randomly select K points as the initial centroid locations'''
    M = points.shape[0] # number of points
    indices = []
    while len(indices) < K:
        index = np.random.randint(0, M)
        if not index in indices:
            indices.append(index)
    initialCentroids = points[indices,:]
    return initialCentroids

def animate(clusterInfo):
    (currentCentroids, classifications, iteration) = clusterInfo
    for k in range(K):
        updatedClusterData = points[classifications == k,:]
        clusterObjList[k].set_data(updatedClusterData[:,0], updatedClusterData[:,1])
        centroidObjList[k].set_data(currentCentroids[k,0], currentCentroids[k,1])
    iterText.set_text('i = {:d}'.format(iteration))

# Create figure and axes. Initialize cluster and centroid line objects.
fig, ax = plt.subplots()
clusterObjList = []
centroidObjList = []
for k in range(K):
    clusterColor = tuple(colorsys.hsv_to_rgb(k / K, 0.8, 0.8))

    clusterLineObj, = ax.plot([], [], ls='None', marker='x', color=clusterColor)
    clusterObjList.append(clusterLineObj)

    centroidLineObj, = ax.plot([], [], ls='None', marker='o',
        markeredgecolor='k', color=clusterColor)
    centroidObjList.append(centroidLineObj)
iterText = ax.annotate('', xy=(0.01, 0.01), xycoords='axes fraction')

def setAxisLimits(ax, points):
    xSpan = np.amax(points[:,0]) - np.amin(points[:,0])
    ySpan = np.amax(points[:,1]) - np.amin(points[:,1])
    pad = 0.05
    ax.set_xlim(np.amin(points[:,0]) - pad * xSpan,
        np.amax(points[:,0]) + pad * xSpan)
    ax.set_ylim(np.amin(points[:,1]) - pad * ySpan,
        np.amax(points[:,1]) + pad * ySpan)

# Initialize data and K-means clustering. Show and animate plot.
points = generateClusters()
initialCentroids = initializeCentroids(K, points)
genFunc = KMeansND(initialCentroids, points).getGeneratorFunc()
setAxisLimits(ax, points)
animObj = animation.FuncAnimation(fig, animate, frames=genFunc,
    repeat=True, interval=500)
plt.ion()
plt.show()

# Construct interactive terminal interface.
inputMessage = ('\nMake a selection:\n'
    + '(1) Randomize clusters and centroids\n'
    + '(2) Randomize centroids only\n'
    + '(3) Save animation to mp4\n'
    + '(4) Exit\n')
while 1:
    if sys.version_info[0] < 3:
        selection = raw_input(inputMessage)
    else:
        selection = input(inputMessage)
    
    if selection == '1':
        animObj._stop()
        print('\nRandomizing clusters and centroids...')
        points = generateClusters()
        initialCentroids = initializeCentroids(K, points)
        genFunc = KMeansND(initialCentroids, points).getGeneratorFunc()
        setAxisLimits(ax, points)
        animObj = animation.FuncAnimation(fig, animate, frames=genFunc,
            repeat=True, interval=500)
    elif selection == '2':
        animObj._stop()
        print('\nRandomizing centroids...')
        initialCentroids = initializeCentroids(K, points)
        genFunc = KMeansND(initialCentroids, points).getGeneratorFunc()
        animObj = animation.FuncAnimation(fig, animate, frames=genFunc,
            repeat=True, interval=500)
        fig.canvas.draw()
    elif selection == '3':
        time = datetime.now()
        timeStr = (str(time.year) + str(time.month) + str(time.day)
            + str(time.hour) + str(time.minute) + str(time.second))
        ffmpegWriterClass = animation.writers['ffmpeg']
        ffmpegWriterObj = ffmpegWriterClass(fps=1, extra_args=['-vcodec', 'h264'])
        filename = timeStr + '_KMeans2D.mp4'
        print('\nSaving file ./' + filename)
        animObj.save(filename, writer=ffmpegWriterObj)
    elif selection == '4':
        exit()
    else:
        print('\nInvalid selection.\n')
