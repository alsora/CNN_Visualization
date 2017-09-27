import numpy as np
import sys

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from skdata.mnist.views import OfficialImageClassification
from skimage.transform import resize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def backspace(n):
    sys.stdout.write('\r'+n)
    sys.stdout.flush()


def binarySearch(comparisonMethod, terms, tolerance=1e-5, objective=30.0):
    valuemin = 0.1

    valuemax = 10000

    value = (valuemin + valuemax) / 2.0

    PP = comparisonMethod(terms, value)

    Pdiff = PP - objective
    binaryTries = 0

    while binaryTries < 100 and np.abs(Pdiff) > tolerance:
        #Try new values until the perplexity difference is under the threshold
        if Pdiff < 0:
            valuemin = value
        else:
            valuemax = value

        value = (valuemin + valuemax) / 2.0

        #Recompute the perplexity
        PP = comparisonMethod(terms, value)

        Pdiff = PP - objective
        binaryTries += 1

    return value


def computePerplexity(D=np.array([]), sigma=1.0):
    #Compute perplexity as function of the gaussian distribution variance sigma

    precision = 1.0 / sigma

    P = np.exp(-D * precision)
    P = P / sum(P)

    log2P = np.log2(P)
    Plog2P = P * log2P
    H = - sum(Plog2P)

    PP = 2 ** H

    return PP


def computeMapPoints(P, numIter=200, numOutputDimensions=2):
    numPoints = P.shape[0]
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 100
    min_gain = 0.01
    #initialize containers with the correct dimension
    gradients = np.zeros((numPoints, numOutputDimensions))
    increment = np.zeros((numPoints, numOutputDimensions))
    gains = np.ones((numPoints, numOutputDimensions))

    mapPointsStorage = []

    #Compute symmetric conditional probabilities (high dimensional space)  
    numeratorPsymmetric = P + np.transpose(P)
    denominatorPsymmetric = np.sum(numeratorPsymmetric)
    Psymmetric = numeratorPsymmetric / denominatorPsymmetric

    Psymmetric *= 4  #Early exaggeration
    #Lower bound on minimum value of high dimensional probabilities
    Psymmetric = np.maximum(Psymmetric, 1e-12) 

    #Initial random low dimensional embedding
    lowDimPoints = np.random.normal(0, 1e-4, (numPoints, numOutputDimensions))

    momentum = initial_momentum
    #T-sne iterations
    for iter in range(numIter):

        mapPointsStorage.append(lowDimPoints)

        #Stop early exaggeration 
        if iter == 100:
            Psymmetric /= 4

        #Switch to the higher momentum after some iterations
        if iter == 250:
            momentum = final_momentum

        #Compute low dimensional pairwise distances
        D = pairwise_distances(lowDimPoints, squared=True)

        #Compute joint probabilities (low dimensional space) Q
        numeratorQ = 1 / (1 + D)
        #Set values on the diagonal to 0
        numeratorQ[range(numPoints), range(numPoints)] = 0 
        denominatorQ = np.sum(numeratorQ)
        Q = numeratorQ/ denominatorQ
        #Lower bound on the minimum value for low dim probabilities
        Q = np.maximum(Q, 1e-12)
        
        #Differences between high and low dimensional probabilities
        P_Q = Psymmetric - Q

        #Compute Kullback-Leibler divergence gradient 
        for i in range(numPoints):
            gradients[i, :] = np.sum(np.tile(P_Q[:, i] * numeratorQ[:, i], (numOutputDimensions, 1)).T * (lowDimPoints[i, :] - lowDimPoints), 0)

        #Select which values have to be increased or decreased
        toBeIncreased = increment * gradients < 0
        toBeDecreased = increment * gradients >= 0
        #Set the corresponding gains
        gains[toBeIncreased] += 0.2
        gains[toBeDecreased] *= 0.8

        gains[gains < min_gain] = min_gain

        learningRate = eta * gains

        increment = momentum * increment - learningRate * gradients

        lowDimPoints = lowDimPoints + increment
        lowDimPoints = lowDimPoints - np.tile(np.mean(lowDimPoints, 0), (numPoints, 1))

        #Compute and display the current cost every 10 iterations
        if (iter + 1) % 10 == 0:
            cost = np.sum(Psymmetric * np.log(Psymmetric / Q))
            to_print = "Iteration {} : error {}".format(iter + 1, cost)
            backspace(to_print)


    #Insert last iteration points
    mapPointsStorage.append(lowDimPoints)
    return mapPointsStorage


def computeProbabilities(X, perplexity=30.0, tolerance=1e-5):
    #Perform an initial dimensionality reduction
    pca = PCA(n_components=50)

    X = pca.fit_transform(X)

    numSamples = X.shape[0]

    P = np.zeros((numSamples, numSamples))

    D = pairwise_distances(X, squared=True)

    for i in range(numSamples):
        indices = np.concatenate((np.arange(i), np.arange(i + 1, numSamples)))

        distancesFromI = D[i, indices]

        sigma = binarySearch(computePerplexity, distancesFromI, tolerance, perplexity)

        precision = 1.0 / sigma
        #Compute a "row" of matrix P: the probabilities wrt point I
        PwrtI = np.exp(- distancesFromI * precision)
        PwrtI /= sum(PwrtI)
        #Insert an element corresponding to I wrt I
        PwrtI = np.concatenate((PwrtI[0:i], [0.0], PwrtI[i:numSamples]))
        #Insert the row
        P[i, :] = PwrtI

    return P


def showPoints(position, labels):
    classes = list(set(labels))

    numClasses = len(classes)

    perClassPositions_t = [[] for x in range(numClasses)]

    for ind, point in enumerate(position):
        subListID = classes.index(labels[ind])

        perClassPositions_t[subListID].append(point)

    finalData = perClassPositions_t
    plotStorage = []
    colors = []

    cmap = plt.cm.get_cmap('hsv', numClasses + 1)

    for index, lab in enumerate(finalData):
        lab = np.asarray(lab)

        x = plt.scatter(lab[:, 0], lab[:, 1], 20, c=cmap(index))

        plotStorage.append(x)
        colors.append(cmap(index))

    plt.legend(plotStorage,
               classes,
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)

    plt.show()


def showMovie(positions, labels):
    positions = np.asarray(positions)

    classes = list(set(labels))

    numClasses = len(classes)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)

    maxX = np.amax(positions[:, :, 0])
    minX = np.amin(positions[:, :, 0])
    maxY = np.amax(positions[:, :, 1])
    minY = np.amin(positions[:, :, 1])

    limit = max(maxX, maxY, minX, minY, key=abs) * 1.2

    ax.set_xlim(-limit, limit), ax.set_xticks([])
    ax.set_ylim(-limit, limit), ax.set_yticks([])
    rect = fig.patch
    rect.set_facecolor('white')

    currentPositions = positions[0]

    colors = []
    cmap = plt.cm.get_cmap('hsv', numClasses + 1)

    for ind in range(numClasses):
        colors.append(cmap(ind))

    coloredLabels = [colors[classes.index(label)] for label in labels]

    scat = ax.scatter(currentPositions[:, 0], currentPositions[:, 1], 20, coloredLabels)

    def increment(frame_number):
        num = frame_number * 5

        currentPositions = positions[num]

        scat.set_offsets(currentPositions)

    numFrames = len(positions) / 5

    animation = FuncAnimation(fig, increment, interval=100, frames=numFrames, repeat=False)
    plt.show()
    animation.save('movie.mp4')



def imagesPlot(images, positions, zoom=0.25):
    fig, ax = plt.subplots()

    for num in range(len(images)):

        x = positions[num, 0]
        y = positions[num, 1]
        image = images[num]

        im = OffsetImage(image, zoom=zoom)
        x, y = np.atleast_1d(x, y)

        for x0, y0 in zip(x, y):
            ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
            ax.add_artist(ab)

        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()

    plt.show()

def test():
    X = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt("mnist2500_labels.txt")

    perplexity = 20.0
    tolerance = 1e-5

    iterations = 800

    P = computeProbabilities(X, perplexity, tolerance)

    positions = computeMapPoints(P, iterations)

    showPoints(positions[-1], labels)

    showMovie(positions, labels)


if __name__ == "__main__":
    print 'Running t-sne test example'
    test()
