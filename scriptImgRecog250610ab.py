# ASDc: Import necessary libraries.
# NOTE: That the Python Imaging Library (PIL) adds image processing capabilities to a Python Interpreter. It is installed via pip as pip install pillow.
from PIL import Image
import os
import numpy
import time
import libImageProcessing
import torch

# ASDc: Clear terminal messages.
os.system('cls' if os.name == 'nt' else 'clear')

# DEFINE CONSTANTS ----- ----- ----- ----- -----

# ASDc: Define the maximum and minimum desired learning rates.
# NOTE: That algorithm begins with maximum and reduced to minimum as needed.
alphaLearnRateMax = 0.5
alphaLearnRateMin = 0.05
alphaLearnRateK = alphaLearnRateMax; # to be used for variable learning rate

# ASDc: Define number of horizontal and vertical sensors for image feature extraction.
# NOTE: That number of neural network inputs will equal numSensorsHorizontal * numSensorsVertical.
numSensorsHorizontal = 4
numSensorsVertical = 4

# ASDc: Calculate number of sensors as well as length of training vector, per sample.
numSensors = numSensorsHorizontal * numSensorsVertical
lenTrainVector = numSensors + 1

# ASDc: Define number of NN outputs, each corresponding to a single digit (0-9).
numOutputs = 10

# ASDc: Define number of previous iterations to be checked for convergence.
lenOfConvergenceCheck = 100

# ASDc: Define maximum loss required for convergence, as well as maximum number of iterations/epochs allowed.
# NOTE: That without adapting learning rates, weight decay, and momentum, it is difficult to drive down the max-convergence value without causing overfitting.
maxLossForConvergence = 0.025
maxEpochs = 25000

# LOAD TRAINING DATA SET / GENERATRE FROM STORED IMAGES ----- ----- ----- ----- -----

print('Load Training Data...')

# ASDc: Check if tenTrainData.pt exists in current directory; if it does, store location to pathTrainingData.
# NOTE: That this file would have be saved by a previous execution of this script.
pathTrainingData = os.path.join(os.path.dirname(__file__), "tenTrainData.pt")

# ASDc: IF (the file exists) THEN...
if os.path.exists(pathTrainingData):

    # ASDc: Load tenTrainData.pt.
    tenTrainData = torch.load(pathTrainingData)

# OTHERWISE...
else:

    # ASDc: Get path of current script.
    scriptPath = __file__

    # ASDc: Get parent (and grandparent) directories.
    dirParentRemoved1 = os.path.dirname(scriptPath) # parent
    dirParentRemoved2 = os.path.dirname(dirParentRemoved1) # grandparent

    # ASDc: Append "SampleImages" and "TrainData" to dirParentRemoved2, identfying the directory in which training images are stored.
    dirTrainData = os.path.join(dirParentRemoved2, "SampleImages")
    dirTrainData = os.path.join(dirTrainData, "TrainData")

    # ASDc: Create array containing names of sub-folders.
    arrSubDirectories = [name for name in os.listdir(dirTrainData) if os.path.isdir(os.path.join(dirTrainData, name))]

    # ASDc: Scan dirTrainData and its sub-directories to quantify number of files (which corresponds to number of training samples).
    numTrainingSamples = sum(
        len([name for name in os.listdir(os.path.join(dirTrainData, subdir)) if os.path.isfile(os.path.join(dirTrainData, subdir, name))])
        for subdir in arrSubDirectories
    )

    # ASDc: Initialize tensor to store all training data, with an ideal label being placed in first column.
    tenTrainData = torch.zeros((numTrainingSamples, lenTrainVector), dtype=torch.float32)

    # ASDc: Cycle through sub-folders within arrSubDirectories.
    for kSubDir, strSubDirectoryK in enumerate(arrSubDirectories):

        # ASDc: Append strSubDirectoryK (kth element of arrSubDirectories) to directory containing all training data.
        pathDirectoryK = os.path.join(dirTrainData, strSubDirectoryK)

        # ASDc: Identify ideal output associated with this sub-directory, as indicated by its name.
        intIdealOutputK = int(strSubDirectoryK)    

        # ASDc: Create array listing all files within pathDirectoryK.
        arrFilesWithinDirectoryK = [name for name in os.listdir(pathDirectoryK) if os.path.isfile(os.path.join(pathDirectoryK, name))]
        
        # ASDc: Create array containing names of files within sub-folder.
        arrFiles = [name for name in os.listdir(pathDirectoryK) if os.path.isfile(os.path.join(pathDirectoryK, name))]
        
        # ASDc: Cycle through files within sub-folder.
        for kFile, strFileK in enumerate(arrFiles):

            # ASDc: Create path to target file (for iteration kFile) by joining directory with file name.
            pathFileK = os.path.join(pathDirectoryK, strFileK)
                
            # ASDc: Use genImageVector() function from libImageProcessing library to generate a vectorization for the image located at pathFileK.
            # NOTE: That the number of horizontal and vertical sensors must be supplied to genImageVector().
            arrImgVectorK = libImageProcessing.genImageVector(pathFileK, numSensorsHorizontal, numSensorsVertical)

            # ASDc: Calculate index of training sample to be assigned.
            kTrainSample = kSubDir * len(arrFiles) + kFile

            # ASDc: Assign ideal output to column-0 of training sample.
            tenTrainData[kTrainSample, 0] = intIdealOutputK
        
            # ASDc: Assign image vector to remaining columns of training sample.
            tenTrainData[kTrainSample, 1:len(arrImgVectorK)+1] = torch.tensor(arrImgVectorK, dtype=torch.float32)

    # ASDc: Save tenTrainData to current directory.
    pathTrainingData = os.path.join(os.path.dirname(scriptPath), "tenTrainData.pt")
    torch.save(tenTrainData, pathTrainingData)
    
# LOAD TEST DATA SET / GENERATRE FROM STORED IMAGES ----- ----- ----- ----- -----

print('Load Test Data...')

# ASDc: Check if tenTestData.pt exists in current directory; if it does, store location to pathTestData.
# NOTE: That this file would have be saved by a previous execution of this script.
pathTestData = os.path.join(os.path.dirname(__file__), "tenTestData.pt")

# ASDc: IF (the file exists) THEN...
if os.path.exists(pathTestData):

    # ASDc: Load tenTestData.pt.
    tenTestData = torch.load(pathTestData)
    
# OTHERWISE...
else:

    # ASDc: Get path of current script.
    scriptPath = __file__

    # ASDc: Get parent (and grandparent) directories.
    dirParentRemoved1 = os.path.dirname(scriptPath) # parent
    dirParentRemoved2 = os.path.dirname(dirParentRemoved1) # grandparent

    # ASDc: Append "SampleImages" and "TestData" to dirParentRemoved2, identfying the directory in which training images are stored.
    dirTestData = os.path.join(dirParentRemoved2, "SampleImages")
    dirTestData = os.path.join(dirTestData, "TestData")

    # ASDc: Create array containing names of sub-folders.
    arrSubDirectories = [name for name in os.listdir(dirTestData) if os.path.isdir(os.path.join(dirTestData, name))]
    
    # ASDc: Scan dirTestData and its sub-directories to quantify number of files (which corresponds to number of training samples).
    numTestSamples = sum(
        len([name for name in os.listdir(os.path.join(dirTestData, subdir)) if os.path.isfile(os.path.join(dirTestData, subdir, name))])
        for subdir in arrSubDirectories
    )

    # ASDc: Initialize tensor to store all test data, with ideal label being placed in first column.
    tenTestData = torch.zeros((numTestSamples, lenTrainVector), dtype=torch.float32)

    # ASDc: Cycle through sub-folders within arrSubDirectories.
    for kSubDir, strSubDirectoryK in enumerate(arrSubDirectories):

        # ASDc: Append strSubDirectoryK (kth element of arrSubDirectories) to directory containing all test data.
        pathDirectoryK = os.path.join(dirTestData, strSubDirectoryK)

        # ASDc: Identify ideal output associated with this sub-directory, as indicated by its name.
        intIdealOutputK = int(strSubDirectoryK)    

        # ASDc: Create array listing all files within pathDirectoryK.
        arrFilesWithinDirectoryK = [name for name in os.listdir(pathDirectoryK) if os.path.isfile(os.path.join(pathDirectoryK, name))]
                
        # ASDc: Create array containing names of files within sub-folder.
        arrFiles = [name for name in os.listdir(pathDirectoryK) if os.path.isfile(os.path.join(pathDirectoryK, name))]
        
        # ASDc: Cycle through files within sub-folder.
        for kFile, strFileK in enumerate(arrFiles):

            # ASDc: Create path to target file (for iteration kFile) by joining directory with file name.
            pathFileK = os.path.join(pathDirectoryK, strFileK)                
            
            # ASDc: Use genImageVector() function from libImageProcessing library to generate a vectorization for the image located at pathFileK.
            # NOTE: That the number of horizontal and vertical sensors must be supplied to genImageVector().
            arrImgVectorK = libImageProcessing.genImageVector(pathFileK, numSensorsHorizontal, numSensorsVertical)
            
            # ASDc: Calculate index of test sample to be assigned.
            kTestSample = kSubDir * len(arrFiles) + kFile

            # ASDc: Assign ideal output to column-0 of test sample.
            tenTestData[kTestSample, 0] = intIdealOutputK
        
            # ASDc: Assign image vector to remaining columns of test sample.
            tenTestData[kTestSample, 1:len(arrImgVectorK)+1] = torch.tensor(arrImgVectorK, dtype=torch.float32)

    # ASDc: Save tenTestData to current directory.
    pathTrainingData = os.path.join(os.path.dirname(scriptPath), "tenTestData.pt")
    torch.save(tenTestData, pathTrainingData)
    
# DEFINE MACHINE LEARNING MODEL ----- ----- ----- ----- -----

class NeuralNetworkForImageRecognition(torch.nn.Module):

    # INITIALIZATION METHOD ----- ----- ----- ----- ----- 
    def __init__(self, numSensorsIN, numOutputsIN):

        # ASDc: Define constructor that builds model with user-defined number of inputs and outputs.
        # NOTE: That this function is always automatically executed every time the class is instantiated (e.g. initiate parameters).
        super(NeuralNetworkForImageRecognition, self).__init__()

        # ASDc: Define two module layers:
        #  1) hidden layer with #numSensorsIN inputs (observed externally) and user-defined number of outputs (to be supplied to next layer).
        #  2) output layer with corresponding number of inputs (supplied by hidden layer) and numOutputsIN outputs (that represent classifications).
        self.hiddenLayer = torch.nn.Linear(numSensorsIN, 2*numSensorsIN)              
        self.outputLayer = torch.nn.Linear(2*numSensorsIN, numOutputsIN)
        
        # ASDc: Define two useful activation functions.
        # NOTE: That ReLU will be applied to hidden layer and sigmoid will be applied to output layer.
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    # FORWARD PROPAGATION METHOD ----- ----- ----- ----- ----- 
    def forward(self, tenInputX):

        # ASDc: Define forward propagation through hidden layer.
        tenY1 = self.relu(self.hiddenLayer(tenInputX))
        
        # ASDc: Define forward propagation through output layer.
        tenOutY2 = torch.sigmoid(self.outputLayer(tenY1))
        
        # ASDc: Return output.
        return tenOutY2
    
# CONFIGURE/INITIALIZE NEURAL NETWORK ----- ----- ----- ----- -----

print('Configure/initialize neural network...')

# ASDc: Instantiate model as neural network with #numSensor inputs and #numOutputs outputs.
myModel = NeuralNetworkForImageRecognition(numSensors, numOutputs)

# ASDc: Instantiate loss function as mean-squared error.
myLossFunction = torch.nn.MSELoss()

# ASDc: Instantiate optimizer with learning rate of alphaLearnRateMax.
# NOTE: That momentum is a parameter that helps our model converge faster and more smoothly by incorporating "memory" of past gradients with 0 representing no momentum 
# and 1 representing complete reliance on momentum  (aka. direction of gradient descent cannot change).
# NOTE: That weight decay represents a form of L2 regularization, discouraging large weights in a model by adding a penalty to the loss function. Relatively small values 
# are used to ensure that decay does NOT prevent meaningful learning.

# NOTE: That weight_decay may be an interesting parameter to use in case where convergence cannot be achieved.
myOptimizer = torch.optim.SGD(myModel.parameters(), lr=alphaLearnRateMax, momentum=0.9, weight_decay=0.0001)

# ASDc: Define test input tensor (with #numSensors inputs).
tenTestInput = torch.ones(numSensors)

# ASDc: Initialize array to store loss values for convergence check.
# NOTE: That lenOfConvergenceCheck determines how many previous values are checked.
arrLoss = numpy.full(lenOfConvergenceCheck, 999, dtype=numpy.float32)

# PERFORM ITERATIVE MACHINE LEARNING ----- ----- ----- ----- -----

print('Begin iterative training/machine learning...')

# ASDc: Begin iterations for machine learning, with kEpoch as the index.
for kEpoch in range(maxEpochs):

    # ASDc: Zero out gradients to prevent value accumulation.
    myOptimizer.zero_grad()

    # ASDc: Determine which training data sample will be used for kIt.    
    kTrain = int(numpy.floor(numpy.random.rand()*(tenTrainData.size(dim=0)))) # choose random element of tenTrainData
    
    # ASDc: Extract training inputs (xTrain) as well as ideal output (correctCat) associated with sample kTrain.
    xTrainK = tenTrainData[kTrain, 1:len(tenTrainData[0,:])]
    correctCat = tenTrainData[kTrain, 0]
    
    # ASDc: Decode correctCat to yield yIdealK.
    yIdealK = torch.zeros(numOutputs, dtype=torch.float32)
    yIdealK[int(correctCat)] = 1.0
        
    # ASDc: Execute forward propagation, generate calculated output.
    yCalcK = myModel(xTrainK)    

    # ASDc: Save current lossK value (before update) to lossKmin1 (aka. K-1).
    # NOTE: That 999 simply indicates that no previous value exists.
    if kEpoch > 0:
        lossKmin1 = lossK
    else:
        lossKmin1 = 999

    # ASDc: Calculate error between ideal and predicted output.
    lossK = myLossFunction(yCalcK, yIdealK)

    # IMPLEMENT ADAPTIVE LEARN RATE ----- ----- ----- ----- ----- 

    # ASDc: IF {loss increases from last iteration to this AND learn rate hasn't reached its allowed minimum} THEN {decrease learning rate by X%}
    if (lossK > lossKmin1) and (alphaLearnRateK > alphaLearnRateMin):
        alphaLearnRateK = 0.9*alphaLearnRateK

        # ASDc: Update optimizer learning rate to current alphaLearnRateK.
        for param_group in myOptimizer.param_groups:
            param_group['lr'] = alphaLearnRateK
            print("Decrease Learning Rate: ", kEpoch, lossK, lossKmin1, alphaLearnRateK)

    # ASDc: IF {loss decreases as expected AND learn rate hasn't reached its allowed maximum} THEN {increase learning rate by X%}
    elif (lossK < lossKmin1) and (alphaLearnRateK < alphaLearnRateMax):
        alphaLearnRateK = 1.1*alphaLearnRateK

        # ASDc: Update optimizer learning rate to current alphaLearnRateK.
        for param_group in myOptimizer.param_groups:
            param_group['lr'] = alphaLearnRateK
            print("Increase Learning Rate: ", kEpoch, lossK, lossKmin1, alphaLearnRateK)

    # ASDc: Store lossK values to an array that constantly repeats, overwriting itself.
    arrLoss[kEpoch%lenOfConvergenceCheck] = lossK.item()
    
    # ASDc: Check for convergence, cease iterations if achieved.
    if(numpy.max(arrLoss)<maxLossForConvergence):

        # ASDc: Save model parameters to tenTrainParams tensor.
        tenTrainedParams = {}
        tenTrainedParams['hiddenLayer.weight'] = myModel.hiddenLayer.weight.data.clone()
        tenTrainedParams['hiddenLayer.bias'] = myModel.hiddenLayer.bias.data.clone()
        tenTrainedParams['outputLayer.weight'] = myModel.outputLayer.weight.data.clone()
        tenTrainedParams['outputLayer.bias'] = myModel.outputLayer.bias.data.clone()

        # ASDc: Break for-loop (because of convergence).
        break
           
    # ASDc: Computes gradient of current tensor (lossK) with respect to graph leaves.
    lossK.backward()

    # ASDc: Perform single optimization step, updating synapse weights to reduce lossK.
    myOptimizer.step()
    
    if(numpy.random.rand()<0.1):
        print('Iteration: ', kEpoch, '; Training Data Sample: ', kTrain)

# TEST TRAINED MODEL ----- ----- ----- ----- ----- 

print('Test trained model...')

# ASDc: Initialize array to contain arrayCompareResults comparison.
arrayCompareResults = numpy.zeros((tenTestData.shape[0], 2), dtype=int)

# ASDc: Cycle through test data samples.
for kTestSample in range(tenTestData.shape[0]):

    # ASDc: Load inputs (xTest), true/ideal output (yTrue), and array output predicted by trained model (yPred).
    xTest = tenTestData[kTestSample, 1:]
    yTrue = int(tenTestData[kTestSample, 0])
    yPred = myModel(xTest)

    # ASDc: Convert array output yPred to a single predicted classification.
    if (torch.argmax(yPred).item() == yTrue):
        if (yPred[yTrue]>0.5):
            predictedClass=yTrue
        else:
            predictedClass=-1
    else:
        if (yPred[torch.argmax(yPred).item()]>0.5):
            predictedClass=torch.argmax(yPred).item()
        else:
            predictedClass=-1

    # ASDc: Save predicted and true/ideal outputs to arrayCompareResults.
    arrayCompareResults[kTestSample,:] = [predictedClass, yTrue]

# ASDc: Calculate accuracy of training via comparison of columns of arrayCompareResults.
accuracy = numpy.mean(arrayCompareResults[:, 0] == arrayCompareResults[:, 1]) * 100
print('Test Accuracy: {:.2f}%'.format(accuracy))
print('Test Results: ', numpy.transpose(arrayCompareResults))