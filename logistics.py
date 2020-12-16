# Logistic Regression with SGD

from pyspark import SparkContext
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint


# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split('')]
    return LabeledPoint(values[0], values[1:])


sc = SparkContext(appName="logistics")
data = sc.textFile("/user/cloudera/sample_svm_data.txt")
parseData = data.map(parsePoint)

# Build the model
model = LogisticRegressionWithSGD.train(parseData)

# Evaluating the model on training data
labelsAndPreds = parseData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda v, p: v != p).count() / float(parseData.count())

print("Training Error = " + str(trainErr))