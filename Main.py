import _pickle
import gzip
import numpy as np
import random

def activation(netInput):
    NUM_OF_ELEMENTS = 10

    for it in range(0, NUM_OF_ELEMENTS):
        if 0 > netInput[it]:
            netInput[it] = 0

    return netInput

def targetToArray(target):
    NUM_OF_PERCEPTRONS = 10

    output = [0 for it in range(0, NUM_OF_PERCEPTRONS)]

    output[target - 1] = 1

    return np.array(output)

def isExpectedOutputReturned(netInput, target):
    returnStatus = True
    maxPerceptronValue = 0
    maxPerceptronIndex = 0
    index = 0

    for value in netInput:
        index = index + 1

        if value > maxPerceptronValue:
            maxPerceptronValue = value
            maxPerceptronIndex = index

    if maxPerceptronIndex != target:
        returnStatus = False

    return returnStatus

def onlineTraining(perceptrons, bias, trainingSet):
    currentIteration = 0
    allClassified = False
    NUM_OF_ITERATIONS = 500
    NUM_OF_IMAGES = 50000
    NUM_OF_PIXELS = 784
    NUM_OF_PERCEPTRONS = 10
    INPUT_VALUE_INDEX = 0
    TARGET_INDEX = 1
    ETA = 0.05

    print('Online Training...')

    while (not allClassified) and (NUM_OF_ITERATIONS > currentIteration):
        allClassified = True

        for it in range(0, NUM_OF_IMAGES):
            inputValue = trainingSet[INPUT_VALUE_INDEX][it]
            target = trainingSet[TARGET_INDEX][it]

            netInput = np.add(np.dot(perceptrons, inputValue), bias)

            netInput = activation(netInput)

            expectedOutput = targetToArray(target)

            biasAdjustment = np.subtract(expectedOutput, netInput)
            biasAdjustment = np.multiply(biasAdjustment, ETA)

            perceptronsAdjustment = np.dot(biasAdjustment.reshape(NUM_OF_PERCEPTRONS, 1), inputValue.reshape(1, NUM_OF_PIXELS))

            perceptrons = np.add(perceptrons, perceptronsAdjustment)

            bias = np.add(bias, biasAdjustment)

            if not isExpectedOutputReturned(netInput, target):
                allClassified = False

        currentIteration = currentIteration + 1

        if allClassified:
            print('All classified!')

    return perceptrons, bias

def test(perceptrons, bias, testSet):
    NUM_OF_IMAGES = 10000
    INPUT_VALUE_INDEX = 0
    TARGET_INDEX = 1
    LOGS_FILE_PATH = './Logs.txt'

    with open(LOGS_FILE_PATH, 'a') as fd:
        for it in range(0, NUM_OF_IMAGES):
            inputValue = testSet[INPUT_VALUE_INDEX][it]
            target = testSet[TARGET_INDEX][it]

            netInput = np.add(np.dot(perceptrons, inputValue), bias)

            maxPerceptronValue = 0
            maxPerceptronIndex = 0
            index = 0

            for perceptron in netInput:
                index = index + 1

                if perceptron > maxPerceptronValue:
                    maxPerceptronValue = perceptron
                    maxPerceptronIndex = index

            fd.writelines(['Expected Digit: ', str(target), 'Neuronal Network Result: ', str(maxPerceptronIndex), '/n'])

    return None

def main():
    NUM_OF_PERCEPTRONS = 10
    NUM_OF_PIXELS = 784

    perceptrons = np.array([[random.random() for col in range(0, NUM_OF_PIXELS)] for row in range(0, NUM_OF_PERCEPTRONS)])
    bias = np.array([random.random() for it in range(0, NUM_OF_PERCEPTRONS)])

    with gzip.open('mnist.pkl.gz', 'rb') as handler:
        trainingSet, validationSet, testSet = _pickle.load(handler, encoding = 'latin1')

        perceptrons, bias = onlineTraining(perceptrons, bias, trainingSet)

        test(perceptrons, bias, testSet)

        handler.close()

    return None

if __name__ == "__main__": main()
