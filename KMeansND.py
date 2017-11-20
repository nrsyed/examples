import numpy as np

def assignPointsToCentroids(centroids, points):
    '''Determine the centroid to which each point is nearest, and
        store this as an int from 0 to K-1 in classifications.
    '''
    M = points.shape[0]
    K = centroids.shape[0]
    classifications = np.zeros((M,), dtype=np.int)

    for i in range(M):
        smallestDistance = 0
        for k in range(K):
            distance = np.linalg.norm(points[i,:] - centroids[k,:])
            if k == 0:
                smallestDistance = distance
                classifications[i] = k
            elif distance < smallestDistance:
                smallestDistance = distance
                classifications[i] = k
    return classifications
        
def recalcCentroids(centroids, points, classifications):
    '''Recalculate centroid locations for each cluster.'''
    K = centroids.shape[0]
    N = points.shape[1]
    M = points.shape[0]

    newCentroids = np.zeros((K, N))
    for k in range(K):
        if sum(classifications == k) > 0:
            newCentroids[k,:] = (
                np.sum(points[classifications == k,:], axis=0)
                / sum(classifications == k))
        else:
            newCentroids[k,:] = centroids[k,:]
    return newCentroids
    
class KMeansND:
    '''KMeansND(initialCentroids, points)

    PARAMETERS:

    initialCentroids: K x N array of K initial centroids with N
        features/coordinates.

    points: M x N array of M points with N features/coordinates.

    METHODS:

    (centroids, classifications, iterations) = getCentroids()
        Perform K-means clustering. Return a tuple containing the
        array of centroid coordinates, an M x 1 array of point
        classifications, and number of iterations required.

    getGenerator()
        Return a generator function to step through K-means iterations.
        Each call of the generator returns the current centroids,
        classifications, and iteration, beginning with the initial
        centroids and classifications.
    '''
    def __init__(self, initialCentroids, points):
        if initialCentroids.shape[1] != points.shape[1]:
            raise RuntimeError('Dimension mismatch. Centroids and data points'
                + ' must be described by the same number of features.')
        else:
            self.initialCentroids = initialCentroids
            self.points = points

    def getCentroids(self):
        centroids = np.copy(self.initialCentroids)
        # Initialize lastCentroids to arbitrary value different from centroids
        # to ensure loop executes at least once.
        lastCentroids = centroids + 1
        iteration = 0
        while not np.array_equal(centroids, lastCentroids):
            lastCentroids = np.copy(centroids)
            classifications = assignPointsToCentroids(centroids, self.points)
            centroids = recalcCentroids(centroids, self.points, classifications)
            iteration += 1
        return (centroids, classifications, iteration)

    def _generatorFunc(self):
        centroids = np.copy(self.initialCentroids)
        lastCentroids = centroids + 1
        iteration = 0
        initialIteration = True
        while not np.array_equal(centroids, lastCentroids):
            if initialIteration:
                classifications = assignPointsToCentroids(centroids, self.points)
                initialIteration = False
            else:
                lastCentroids = np.copy(centroids)
                classifications = assignPointsToCentroids(centroids, self.points)
                centroids = recalcCentroids(centroids, self.points, classifications)
                iteration += 1
            yield (centroids, classifications, iteration)

    def getGeneratorFunc(self):
        return self._generatorFunc
