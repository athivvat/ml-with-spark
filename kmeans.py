# K-Means

from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans
from numpy import array
from math import sqrt

sc = SparkContext(appName="kmeans")

# Load and parse the data
data = sc.textFile("/user/cloudera/kmeans_data.txt")
parsedData = data.map(lambda line: array([float(x) for x in line.split('')]))

# Build the model (cluster the data)
clusters = KMeans.train(parsedData, 2, maxIterations=10, runs=10, initializationMode="random")


# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))


WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)

print("Within Set Sum of Square Error = " + str(WSSSE))