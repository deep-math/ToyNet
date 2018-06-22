%
% mnist_test.m tests ToyNet and perturbation algorithm
%
clear;
[trainImages,trainLabels, validatimages, validatLabels] = loadMNIST('mnist/train-images.idx3-ubyte', 'mnist/train-labels.idx1-ubyte','mnist/t10k-images.idx3-ubyte','mnist/t10k-labels.idx1-ubyte');

load('resources/trainedToyNet_v01');    % Load pretrained ToyNet
load('resources/deepFooledImage');


% Set to true if need to retrain
first_time_launch = false;
doPerturbation = true;

% Training part. Not needed.
if first_time_launch == true
    % Init NN and train it
    tn = ToyNet(2,784,10,20);    % Input params: i_numHiddenLayers, i_inputLayerSize, i_outputLayerSize, i_hiddenLayersSize
    % train(tn, trainImages, trainLabels, 1000000, 0.06);   % slow but accurate training
    disp('training...');
    train(tn, trainImages, trainLabels, 400000, 0.12);      % fast but not accurate training
end

% Pick image then forwardProp image and print result in the console.
index = 33;     % Pick some image by its index (digit 3 is index 33)
testImg =  validatimages(:,index);
[~,digitNumber] = max(validatLabels(:,index))
perturbedImg = testImg;
classifRes = ones(10,1);

noisyImg = min(testImg + 0.2*rand(784,1), 1);   % limit the range from 0 to 1

% Perturbation generation
disp('working...');
while classifRes(digitNumber) > 0.5 && doPerturbation == true
% for i=1:500000
    forwardProp(tn, perturbedImg);
    perturbedImg = adversBackProp(tn, perturbedImg,validatLabels(:,index), 0.7);
    classifRes = forwardProp(tn, perturbedImg);
    classifRes(digitNumber);
end

% Classify images
classificationResultPerturb = forwardProp(tn, perturbedImg)
classificationResultOrig = forwardProp(tn, testImg)
classificationResultNoisy = forwardProp(tn, noisyImg)
deepFoolImg = reshape(dfImg,784,1);
classificationResultDeepFool =  forwardProp(tn, deepFoolImg)


% Didsplay picked image
figure;
subplot(1,4,1);
digitPerturbed = reshape(perturbedImg, [28,28]);    % row = 28 x 28 image
imshow(digitPerturbed*255, [0 255])      % show the image
title('perturbed');

subplot(1,4,2);
digitOrig = reshape(testImg, [28,28]);    % row = 28 x 28 image
imshow(digitOrig*255,[0 255])      % show the image
title('original');
norm(perturbedImg -testImg)

subplot(1,4,3);
digitNoise = reshape(noisyImg, [28,28]);    % row = 28 x 28 image
imshow(digitNoise*255,[0 255])      % show the image
title('random noise');

subplot(1,4,4);
imshow(dfImg*255,[0 255]);      % show the image
title('deep fool perturbation');
