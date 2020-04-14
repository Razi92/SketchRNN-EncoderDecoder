##########################
## libraries
##########################
from DataGeneration import loadAndExport
from DataPreprocessing import DataLoader as dl
from Networks.UNet import UNet
from Utils import computeBatchLoss
from GUI.GraphCreator import createVectorGraph
from DataGeneration.GridPadder import unpadArrays

import numpy as np
import random
import os

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
# auto-gradienimport matplotlib.pyplot as plt


# from DataPreprocessing.Utils import scaleToChannels
# x = np.array([[[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20],[21,22,23,24]]]])
# x = np.array([[[[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18],[19,20,21,22,23,24]]]])
# y = scaleToChannels(x, (2, 2))


##########################
## settings and hyper-parameters
##########################
cudaAvailable = torch.cuda.is_available()
device = torch.device("cuda:0" if cudaAvailable else "cpu")

# filenameLRES = "/home/razi/Documents/3_codesFromGitHub/0_Grib_Windfields/Data_windfields/ERA5_predictand_vars_2018.grib"
# filenameLRES_Sfc = "/home/razi/Documents/3_codesFromGitHub/0_Grib_Windfields/Data_windfields/ERA5_Auxil_Fixed_SurfaceVars.grib"
# filenameHRES = "/home/razi/Documents/3_codesFromGitHub/0_Grib_Windfields/Data_windfields/HRES_100m_wind_cpts_2018.grib"
# filenameHRES_Sfc = "/home/razi/Documents/3_codesFromGitHub/0_Grib_Windfields/Data_windfields/HRES_Auxil_Fixed_SurfaceVars.grib"

#filenameLRES = 'Data_Windfields/ERA5_predictand_vars_2018.grib'
#filenameLRES_Sfc = 'Data_Windfields/ERA5_Auxil_Fixed_SurfaceVars.grib'
#filenameHRES = 'Data_Windfields/HRES_100m_wind_cpts_2018.grib'
#filenameHRES_Sfc = 'Data_Windfields/HRES_Auxil_Fixed_SurfaceVars.grib'

# counting the number of files in the folder starting with letters "run"
def findRunNumber(folder):
    files = os.listdir(folder)
    files = sorted([f for f in files if f.startswith('run')])
    return len(files)

resultsFolder = 'results'
recordsFolder = 'records'
modelsFolder = 'models'

recordsDir = os.path.join(resultsFolder, recordsFolder)
runNumber = findRunNumber(recordsDir)

runName = 'run_%05d' % runNumber
logDir = os.path.join(recordsDir, runName)

if not os.path.exists(logDir):
    os.makedirs(logDir)

modelsDir = os.path.join(logDir, modelsFolder)

if not os.path.exists(recordsDir):
    os.makedirs(recordsDir)

if not os.path.exists(modelsDir):
    os.makedirs(modelsDir)

# outdirOriginal = 'results/original'
# outdirPadding = 'results/padded'

outdirTraining = 'data/training'
outdirTest = 'data/test'

paddingHRES = (180, 144)
paddingLRES = (60, 36)

preprocess = True
inputChannels = 64
numFiles = 8000
learningRate = 0.001
decayLR = True
LRExpFactor = -1
batchSize = 16
iterations = 50000
stepSize = 20
gamma = 0.9

# initialize summary writer
summaryWriter = SummaryWriter(logDir, purge_step=1)
# checkpoint save
saveEveryNthEpoch = 10

# Random seeds, what this seed is used for?
'''seed = random.randint(0, 2**32 - 1) # returns a number such that a <= N <= b
print("Random seed: {}".format(seed)) # .format puts the string or number in () inside the {} automatically and removes {}
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)'''


##########################
## 1) Generate data
##########################
gridsInput = {'windU': '100u', 'windV': '100v', 'blh': 'blh', 'fsr': 'fsr'}
gridsTarget = {'windU': '100u', 'windV': '100v'}
gridsOro = {'z': 'z', 'seaMask': 'lsm'}

if preprocess:

    loadAndExport(filenameLRES, 'input', outdirTraining, outdirTest, (1, 10, 2018), (11, 12, 2018),
                  gridsInput, padding=True, paddingShape=paddingLRES)
    loadAndExport(filenameLRES_Sfc, 'input_orography', outdirTraining, outdirTest, (1, 12, 2018), (1, 12, 2018),
                  gridsOro, padding=True, paddingShape=paddingLRES, formatTime=False)

    loadAndExport(filenameHRES, 'target', outdirTraining, outdirTest, (1, 10, 2018), (11, 12, 2018),
                  gridsTarget, padding=True, paddingShape=paddingHRES)
    loadAndExport(filenameHRES_Sfc, 'target_orography', outdirTraining, outdirTest, (1, 12, 2018), (1, 12, 2018),
                  gridsOro, padding=True, paddingShape=paddingHRES, formatTime=False)

    exit(0)


##########################
## 2) Load data
##########################
opt = dl.vecFieldOptions('input', 'input_orography', ['windU', 'windV', 'seaMask', 'z'],
                         'target', 'target_orography', ['windU', 'windV'],
                         numFiles, False, False,
                         (1, 10), (10, 10), True)

dataTraining, dataValidation = dl.loadVectorFieldsFromDirectory(outdirTraining, opt)
dataTraining.normalize()
dataValidation.normalize()

trainLoader = DataLoader(dataTraining, batch_size=batchSize, shuffle=True, drop_last=True)
validationLoader = DataLoader(dataValidation, batch_size=batchSize, shuffle=True, drop_last=True)
#
# inputs = Variable(torch.FloatTensor(batchSize, dataTraining.inputs.shape[1], dataTraining.inputs.shape[2],
#                                     dataTraining.inputs.shape[3]))
# targets = Variable(torch.FloatTensor(batchSize, dataValidation.targets.shape[1], dataValidation.targets.shape[2],
#                                     dataValidation.targets.shape[3]))
# inputs = inputs.to(device)
# targets = targets.to(device)
#
# inputsVali = Variable(torch.FloatTensor(batchSize, dataValidation.inputs.shape[1], dataValidation.inputs.shape[2],
#                                     dataValidation.inputs.shape[3]))
# targetsVali = Variable(torch.FloatTensor(batchSize, dataValidation.targets.shape[1], dataValidation.targets.shape[2],
#                                     dataValidation.targets.shape[3]))
# inputsVali = inputsVali.cuda()
# targetsVali = targetsVali.cuda()


##########################
## 2) 3) Setup networks, loss, optimizer, and scheduler
##########################


def initWeights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# build network
net = UNet(dataTraining.inputs.shape[1], dataTraining.targets.shape[1], inputChannels)
# randomize weights for initial step
net.apply(initWeights)
print(net)
#print(net.parameters())
# transform to cuda
net.cuda()

# track learnable parameters
modelParameters = filter(lambda p: p.requires_grad, net.parameters())
params = sum([np.prod(p.size()) for p in modelParameters])
#print("Initialized TurbNet with {} trainable params ".format(params))

# create optimizer ADAM with learning rate
optimizer = optim.Adam(net.parameters(), lr=learningRate, betas=(0.5, 0.999), weight_decay=0.0)
# create scheduler to reduce learning rate on-the-fly, lr decay
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=stepSize, gamma=gamma)

###########################
# 4) Setup loss function
lossFunction = nn.L1Loss()
lossFunction.to(device)

###########################
# 5) Setup training
epochs = int(iterations / len(trainLoader) + 0.5)


def saveCheckpoint(epoch):
    modelName = 'UNet'
    modelOutDir = os.path.join(modelsDir, '{}_epoch_{}.pth'.format(modelName, epoch))

    state = {'epoch': epoch, 'model': net}
    state.update({'optimizer': optimizer, 'scheduler': scheduler})

    torch.save(state, modelOutDir)
    print("Saved checkpoint for epoch {} to file {}".format(epoch, modelOutDir))


def train(epoch, epochs):
    # set mode of network to train (parameters will be affected / updated)
    net.train()
    # advance scheduler to perform lr decay after certain steps
    scheduler.step()

    # loss for this epoch
    trainLoss = 0
    trainLossReal = 0
    trainLossUpsampled = 0

    # for p in net.parameters():
    #     p.requires_grad = True

    # iterate over all batches
    for i, trainData in enumerate(tqdm(trainLoader), 0):
        inputs_cpu, targets_cpu, gridLR, gridHR, maskLR, maskHR, idx = trainData
        targets, inputs = targets_cpu.to(device), inputs_cpu.to(device)

        # convert GPU / tensor data to numpy arrays
        inputs_cpu = inputs_cpu.data.numpy()
        targets_cpu = targets_cpu.data.numpy()
        gridsLR = gridLR.data.numpy()
        gridsHR = gridHR.data.numpy()
        masksLR = maskLR.data.numpy()
        masksHR = maskHR.data.numpy()

        # zero out the parameters of the gradients
        optimizer.zero_grad()
        # forward input to the network
        predictions = net(inputs)
        # compute loss to current targets
        lossL1 = lossFunction(predictions, targets)
        # backpropagate loss function
        lossL1.backward()
        # advance weights towards minimum
        optimizer.step()
        # update training loss
        trainLoss += lossL1.item()

        predictions_cpu = predictions.data.cpu().numpy()
        # # denormalize data
        # dataTraining.denormalizeOutput(targets_cpu[0], idx[0])
        # dataTraining.denormalizeOutput(predictions_cpu[0], idx[0])

        # compute real loss
        trainLossReal += computeBatchLoss(predictions_cpu, targets_cpu,
                                               idx, [0, 1], dataTraining)

        # upsample images bilinear
        upsamplesPred = []
        targetSize = targets_cpu.shape[2:4]
        # upsamplesPred = F.interpolate(inputs[:, [0, 1]], targetSize)

        for j in range(len(inputs_cpu)):
            input = inputs_cpu[j]
            index = idx[j]
            dataTraining.denormalizeOutput(input, index, False)

        inputs_temp = torch.tensor(inputs_cpu)
        upsamplesPred = F.interpolate(inputs_temp[:, [0, 1]], size=targetSize, mode='bilinear')
        targets_temp = torch.tensor(targets_cpu)

        lossUpsampledL1 = lossFunction(upsamplesPred, targets_temp).item()
        trainLossUpsampled += lossUpsampledL1

        # sample images
        if i == 0:
            # denormalize data
            input0 = inputs_cpu[0]
            target0 = targets_cpu[0]
            pred0 = predictions_cpu[0]
            predUp = upsamplesPred.data.cpu().numpy()[0]
            gridLR = gridsLR[0]
            gridHR = gridsHR[0]
            maskLR = masksLR[0]
            maskHR = masksHR[0]

            # dataTraining.denormalizeOutput(input0, idx[0], False)
            # dataTraining.denormalizeOutput(predUp, idx[0])

            #predBilinear = F.interpolate(input0, size=paddingHRES, mode='bilinear')

            fig = plt.figure(figsize=(21.6, 21.6))
            #fig = plt.figure(figsize=(25.6, 14.4))
            #fig = plt.figure(figsize=(16.8, 10.5))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title("Input", fontsize=40)
            grids = unpadArrays([input0[0], input0[1], gridLR[0], gridLR[1]], maskLR)
            createVectorGraph(grids[0], grids[1], grids[2], grids[3], scale=3)
            plt.tight_layout()
            summaryWriter.add_figure('train/figInput'.format(epoch), fig, epoch)
            #summaryWriter.add_figure('train/figInput'.format(epoch), inputFig, epoch)
            #plt.close()

            #fig = plt.figure(figsize=(16.8, 10.5))
            fig = plt.figure(figsize=(43.2, 21.6))
            #fig = plt.figure(figsize=(25.6, 14.4))
            gridCur = unpadArrays([gridHR[0], gridHR[1]], maskHR)

            predUV = unpadArrays([pred0[0], pred0[1]], maskHR)
            ax = fig.add_subplot(1, 2, 1)
            ax.set_title("Prediction", fontsize=40)
            createVectorGraph(predUV[0], predUV[1], gridCur[0], gridCur[1])
            #summaryWriter.add_figure('train/figPredict'.format(epoch), targetFig, epoch)
            #plt.close()

            grids = unpadArrays([target0[0], target0[1]], maskHR)
            ax = fig.add_subplot(1, 2, 2)
            ax.set_title("Target", fontsize=40)
            createVectorGraph(grids[0], grids[1], gridCur[0], gridCur[1])
            plt.tight_layout(pad=0, w_pad=0, h_pad=0)
            summaryWriter.add_figure('train/figPredTarget'.format(epoch), fig, epoch)
            # summaryWriter.add_figure('train/figTarget'.format(epoch), targetFig, epoch)
            # plt.close()

            #grids = unpadArrays([target0[0], target0[1]], maskHR)
            fig = plt.figure(figsize=(43.2, 21.6))
            ax = fig.add_subplot(1, 2, 2)
            ax.set_title("Target", fontsize=40)
            createVectorGraph(grids[0], grids[1], gridCur[0], gridCur[1])
            # summaryWriter.add_figure('train/figTarget'.format(epoch), targetFig, epoch)
            # plt.close()

            predUpUV = unpadArrays([predUp[0], predUp[1]], maskHR)
            ax = fig.add_subplot(1, 2, 1)
            ax.set_title("Upsampling (Bilinear)", fontsize=40)
            createVectorGraph(predUpUV[0], predUpUV[1], gridCur[0], gridCur[1])
            #summaryWriter.add_figure('train/figUpBilinear'.format(epoch), targetFig, epoch)
            #plt.close()
            plt.tight_layout(pad=0, w_pad=0, h_pad=0)
            summaryWriter.add_figure('train/figUpTarget'.format(epoch), fig, epoch)

    trainLoss /= len(trainLoader)
    trainLossReal /= len(trainLoader)
    trainLossUpsampled /= len(trainLoader)

    # update records per train step
    summaryWriter.add_scalar('train/lossL1', trainLoss, epoch)
    summaryWriter.add_scalar('train/lossRealL1', trainLossReal, epoch)
    summaryWriter.add_scalar('train/lossUpsampledL1', trainLossUpsampled, epoch)
    summaryWriter.add_scalar('train/lr', scheduler.get_lr()[0], epoch)

    # print("batch-idx: {}".format(i))
    print("Training: L1 {}, LRDecay {}".format(trainLoss, scheduler.get_lr()[0]))
    print("Training: L1 Real {},".format(trainLossReal))
    print("Training: L1 Bilinear {},".format(trainLossUpsampled))

    if epoch % saveEveryNthEpoch == 0:
        saveCheckpoint(epoch)


def validate(epoch, epochs):
    # set mode of network to evaluation to not update any parameters
    net.eval()
    validationLoss = 0
    validationLossReal = 0
    validationLossUpsampled = 0

    for i, validationData in enumerate(validationLoader, 0):
        inputs_cpu, targets_cpu, gridLR, gridHR, maskLR, maskHR, idx = validationData

        targets, inputs = targets_cpu.to(device), inputs_cpu.to(device)

        inputs_cpu = inputs_cpu.data.numpy()
        targets_cpu = targets_cpu.data.numpy()
        gridsLR = gridLR.data.numpy()
        gridsHR = gridHR.data.numpy()
        masksLR = maskLR.data.numpy()
        masksHR = maskHR.data.numpy()

        # predict validation data
        predictions = net(inputs)
        # compute loss to current targets
        lossL1 = lossFunction(predictions, targets)
        # update loss
        validationLoss += lossL1.item()

        predictions_cpu = predictions.data.cpu().numpy()

        # compute real loss
        validationLossReal += computeBatchLoss(predictions_cpu, targets_cpu,
                                               idx, [0, 1], dataValidation)

        targetSize = (paddingHRES[1], paddingHRES[0])
        # upsamplesPred = F.interpolate(inputs[:, [0, 1]], targetSize)

        for j in range(len(inputs_cpu)):
            input = inputs_cpu[j]
            index = idx[j]
            dataTraining.denormalizeOutput(input, index, False)

        inputs_temp = torch.tensor(inputs_cpu)
        upsamplesPred = F.interpolate(inputs_temp[:, [0, 1]], size=targetSize, mode='bilinear')
        targets_temp = torch.tensor(targets_cpu)

        lossUpsampledL1 = lossFunction(upsamplesPred, targets_temp).item()
        validationLossUpsampled += lossUpsampledL1

        # sample images
        if i == 0 and epoch > 20:
            # denormalize data
            input0 = inputs_cpu[0]
            target0 = targets_cpu[0]
            pred0 = predictions_cpu[0]
            predUp = upsamplesPred.data.cpu().numpy()[0]
            gridLR = gridsLR[0]
            gridHR = gridsHR[0]
            maskLR = masksLR[0]
            maskHR = masksHR[0]

            # dataTraining.denormalizeOutput(input0, idx[0], False)
            # dataTraining.denormalizeOutput(predUp, idx[0])

            # predBilinear = F.interpolate(input0, size=paddingHRES, mode='bilinear')

            fig = plt.figure(figsize=(21.6, 21.6))
            # fig = plt.figure(figsize=(25.6, 14.4))
            # fig = plt.figure(figsize=(16.8, 10.5))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title("Input", fontsize=40)
            grids = unpadArrays([input0[0], input0[1], gridLR[0], gridLR[1]], maskLR)
            createVectorGraph(grids[0], grids[1], grids[2], grids[3], scale=3)
            plt.tight_layout()
            summaryWriter.add_figure('validation/figInput'.format(epoch), fig, epoch)
            # summaryWriter.add_figure('train/figInput'.format(epoch), inputFig, epoch)
            # plt.close()

            # fig = plt.figure(figsize=(16.8, 10.5))
            fig = plt.figure(figsize=(43.2, 21.6))
            # fig = plt.figure(figsize=(25.6, 14.4))
            gridCur = unpadArrays([gridHR[0], gridHR[1]], maskHR)

            predUV = unpadArrays([pred0[0], pred0[1]], maskHR)
            ax = fig.add_subplot(1, 2, 1)
            ax.set_title("Prediction", fontsize=40)
            createVectorGraph(predUV[0], predUV[1], gridCur[0], gridCur[1])
            # summaryWriter.add_figure('train/figPredict'.format(epoch), targetFig, epoch)
            # plt.close()

            grids = unpadArrays([target0[0], target0[1]], maskHR)
            ax = fig.add_subplot(1, 2, 2)
            ax.set_title("Target", fontsize=40)
            createVectorGraph(grids[0], grids[1], gridCur[0], gridCur[1])
            plt.tight_layout(pad=0, w_pad=0, h_pad=0)
            summaryWriter.add_figure('validation/figPredTarget'.format(epoch), fig, epoch)
            # summaryWriter.add_figure('train/figTarget'.format(epoch), targetFig, epoch)
            # plt.close()

            # grids = unpadArrays([target0[0], target0[1]], maskHR)
            fig = plt.figure(figsize=(43.2, 21.6))
            ax = fig.add_subplot(1, 2, 2)
            ax.set_title("Target", fontsize=40)
            createVectorGraph(grids[0], grids[1], gridCur[0], gridCur[1])
            # summaryWriter.add_figure('train/figTarget'.format(epoch), targetFig, epoch)
            # plt.close()

            predUpUV = unpadArrays([predUp[0], predUp[1]], maskHR)
            ax = fig.add_subplot(1, 2, 1)
            ax.set_title("Upsampling (Bilinear)", fontsize=40)
            createVectorGraph(predUpUV[0], predUpUV[1], gridCur[0], gridCur[1])
            # summaryWriter.add_figure('train/figUpBilinear'.format(epoch), targetFig, epoch)
            # plt.close()
            plt.tight_layout(pad=0, w_pad=0, h_pad=0)
            summaryWriter.add_figure('validation/figUpTarget'.format(epoch), fig, epoch)

    numLoadings = len(validationLoader)
    validationLoss /= numLoadings
    validationLossReal /= numLoadings
    validationLossUpsampled /= numLoadings

    summaryWriter.add_scalar('validation/lossL1', validationLoss, epoch)
    summaryWriter.add_scalar('validation/lossRealL1', validationLossReal, epoch)
    summaryWriter.add_scalar('validation/lossUpsampledL1', validationLossUpsampled, epoch)

    print("Validation: L1 {}".format(validationLoss))
    print("Validation: L1 Real {},".format(validationLossReal))
    print("Validation: L1 Bilinear {},".format(validationLossUpsampled))

###########################
# 6) Start training

print("===========================")

for epoch in range(epochs):
    print("Epoch {} / {}".format(epoch, epochs))
    train(epoch, epochs)
    validate(epoch, epochs)
    print("===========================")


