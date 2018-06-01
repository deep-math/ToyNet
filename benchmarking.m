%
% benchmarking.m gets ToyNet's performance data
%

clear;
[trainImages,trainLabels, validatimages, validatLabels] = loadMNIST('mnist/train-images.idx3-ubyte', 'mnist/train-labels.idx1-ubyte','mnist/t10k-images.idx3-ubyte','mnist/t10k-labels.idx1-ubyte');

load('resources/trainedToyNet_v01');    % Load pretrained ToyNet

retrain = false;

% Train the network
if retrain == true
    disp('training...');
    % Init NN and train it
    tn = ToyNet(2,784,10,200);    % Input params: i_numHiddenLayers, i_inputLayerSize, i_outputLayerSize, i_hiddenLayersSize
    train(tn, trainImages, trainLabels, 2000000, 0.15);   % slow but accurate training
    % train(tn, trainImages, trainLabels, 400000, 0.12);      % fast but not accurate training
    disp('training complete.');
end

% Training and Validation costs and errors
trainingCost = 0;
generalizationCost = 0;
trainingError = 0;
validationError = 0;
upperBound = 0.95;
lowerBound = 0.2;
disp('benchmarking running...');


trainCycles = 60000;
validCycles = 10000;

% Training benchmarking
for i=1:trainCycles
    %  Compute training cost using 2-norm
    Y = forwardProp(tn, trainImages(:,i));
    L = trainLabels(:,i);
    trainingCost = trainingCost + norm(Y-L)^2;
    % Compute training error
    trainingError = trainingError + computeError(Y, L, upperBound, lowerBound);
end

% Validation benchmarking
for i=1:validCycles
    % Compute validation error
    Y = forwardProp(tn, validatimages(:,i));
    L = validatLabels(:,i);
    generalizationCost = generalizationCost + norm(Y-L)^2;

    validationError = validationError + computeError(Y, L, upperBound, lowerBound);

    % [resProbability,resLabel] = max(Y);
end

trainingCost = trainingCost / trainCycles;
trainingError = trainingError / trainCycles;
generalizationCost = generalizationCost / validCycles;
validationError = validationError / validCycles;

result = ['TrainingCost: ', num2str(trainingCost, '%0.2f'),' GeneralizationCost: ', num2str(generalizationCost, '%0.2f'),' TrainingError: ', num2str(trainingError,'%0.2f'), ' ValidationError: ', num2str(validationError, '%0.2f')];

disp(result);

% Binary function. If classied label does not correspond to the actual label return 1.
% If classified label is correct but classification probability is below treshold or any other classified label greater than 1 - treshold return 1
function error = computeError(predictionVec, labelVec, upperBound, lowerBound)
    [~,label] = max(labelVec);
    [predictions, labelsIndices] = sort(predictionVec, 'descend');

    if labelsIndices(1) ~= label
        error = 1.0;
    elseif predictionVec(labelsIndices(1)) < upperBound || predictionVec(labelsIndices(2)) > lowerBound
        error = 1.0;
    else
        error = 0.0;
    end
end
