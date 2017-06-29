
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from skdata.mnist.views import OfficialImageClassification





def binarySearch(comparisonMethod, terms, tolerance = 1e-5, objective = 30.0):

	valuemin = 0.1

	valuemax =  10000

	value = (valuemin + valuemax)/2.0


	PP = comparisonMethod(terms, value)

	Pdiff = PP - objective
	tries = 0

	while tries < 50 and np.abs(Pdiff) > tolerance:
		# If not, increase or decrease precision
		if Pdiff < 0:
			valuemin = value
		else:
			valuemax = value

		value = (valuemin + valuemax)/2.0

		# Recompute the values
		PP = comparisonMethod(terms, value)

		Pdiff = PP - objective
		tries = tries + 1



	return value


def computeRowP(D = np.array([]), sigma = 1.0):
	precision = 1.0/sigma

	P = np.exp(-D * precision)
	P = P / sum(P)

	return P



def computePerplexity(D = np.array([]), sigma = 1.0):
	"""Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

	# Compute P-row and corresponding perplexity
	precision = 1.0/sigma

	P = np.exp(-D * precision)
	P = P / sum(P)
	
	log2P = np.log2(P)
	Plog2P = P*log2P
	H = - sum(Plog2P)
	
	PP = 2**H	

	return PP



def computeMapPoints(P, max_iter = 200, no_dims = 2):
	n = P.shape[0]
	initial_momentum = 0.5
	final_momentum = 0.8
	eta = 100
	min_gain = 0.01
	Y = np.random.normal(0, 1e-4, (n, no_dims))
	grad = np.zeros((n, no_dims))
	update = np.zeros((n, no_dims))
	gains = np.ones((n, no_dims))

	mapPointsStorage = []

	P = P + np.transpose(P)
	P = P / np.sum(P)
	P = P * 4							# early exaggeration
	P = np.maximum(P, 1e-12)


	# Run iterations
	for iter in range(max_iter):

		mapPointsStorage.append(Y)

		# Compute pairwise affinities
		

		D = pairwise_distances(Y, squared=True)

		num = 1/(1 + D)

		num[range(n), range(n)] = 0
		Q = num / np.sum(num)
		Q = np.maximum(Q, 1e-12)

		# Compute gradient
		PQ = P - Q

		for i in range(n):
			grad[i,:] = np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0)

		# Perform the update
		if iter < 250:
			momentum = initial_momentum
		else:
			momentum = final_momentum

		
		toBeIncreased = update*grad < 0
		toBeDecreased = update*grad >= 0
		gains[toBeIncreased] += 0.2	
		gains[toBeDecreased] *= 0.8


		gains[gains < min_gain] = min_gain

		learningRate = eta*gains


		update = momentum * update - learningRate * grad

		Y = Y + update
		Y = Y - np.tile(np.mean(Y, 0), (n, 1))

		# Compute current value of cost function
		if (iter + 1) % 10 == 0:
			C = np.sum(P * np.log(P / Q))
			print "Iteration ", (iter + 1), ": error is ", C

		# Stop early exaggeration
		if iter == 100:
			P = P / 4


	mapPointsStorage.append(Y)
	return mapPointsStorage


def computeProbabilities(X, perplexity = 30.0, tolerance = 1e-5):

	pca = PCA(n_components=50) 

	X = pca.fit_transform(X)

	numSamples = X.shape[0]

	P = np.zeros((numSamples, numSamples))

	D = pairwise_distances(X, squared=True)

	for i in range(numSamples):



		indices = np.concatenate((np.arange(i), np.arange(i+1,numSamples)))

		distancesFromI = D[i,indices]

		sigma = binarySearch(computePerplexity, distancesFromI, tolerance, perplexity)

		Prow = computeRowP(distancesFromI, sigma)

		P[i,:] = np.concatenate((Prow[0:i],[0.0],Prow[i:numSamples]))



	return P






def showPoints(positions, labels, movie=True):

	finalData = positions[-1] 
	plt.scatter(finalData[:,0], finalData[:,1], 20, labels)
	plt.show();

	positions = np.asarray(positions)

	if movie:
		fig = plt.figure(figsize=(10, 10))
		ax = fig.add_axes([0, 0, 1, 1], frameon=False)

		maxX = np.amax(positions[:,:,0])
		minX = np.amin(positions[:,:,0])
		maxY = np.amax(positions[:,:,1])
		minY = np.amin(positions[:,:,1])

		limit = max(maxX, maxY, minX, minY, key=abs) * 1.2

		ax.set_xlim(-limit, limit), ax.set_xticks([])
		ax.set_ylim(-limit, limit), ax.set_yticks([])
		rect = fig.patch
		rect.set_facecolor('white')

		currentPositions = positions[0]

		scat = ax.scatter(currentPositions[:, 0], currentPositions[:, 1],20, labels )

		def update(frame_number):

		    num = frame_number*5

		    currentPositions = positions[num]

		    scat.set_offsets(currentPositions)


		# Construct the animation, using the update function as the animation
		# director.

		numFrames = len(positions)/5

		animation = FuncAnimation(fig, update, interval=100, frames = numFrames, repeat = False)
		plt.show()
		animation.save('movie.mp4')




def test():
	X = np.loadtxt("mnist2500_X.txt")
	labels = np.loadtxt("mnist2500_labels.txt")

	perplexity = 20.0
	tolerance = 1e-5

	iterations = 800
	
	P = computeProbabilities(X, perplexity, tolerance)

	positions = computeMapPoints(P, iterations)

	showPoints(positions, labels)




if __name__ == "__main__":
	print 'Running t-sne test example'
	test()
	




