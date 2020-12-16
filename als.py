from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
from numpy import array

sc = SparkContext(appName='als')

# Load and parse the data
data = sc.textFile("/user/cloudera/test.data")
ratings = data.map(lambda line: line.split(',')).map(lambda r: (int(r[0]), int(r[1]), float(r[2])))

# Build the recommendation model using Alternating Least Squares
rank = 10
numIterations = 20
model = ALS.train(ratings, rank, numIterations)

# Evaluate the model on training data
testdata = ratings.map(lambda p: (p[0], p[1]))
pd = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
predictions = pd.sortByKey().map(lambda r: r[1])
rd = ratings.map(lambda r: ((r[0], r[1]), r[2]))
actualRates = rd.sortByKey().map(lambda r: r[1])

ratesAndPreds = actualRates.zip(predictions)

MSE = ratesAndPreds.map(lambda r: (r[0] - r[1])**2).reduce(lambda x, y: x + y/ratesAndPreds.count())
print("Mean Squared Error = " + str(MSE))