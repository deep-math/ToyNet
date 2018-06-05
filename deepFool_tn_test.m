%
% mnist_test.m tests ToyNet and perturbation algorithm
%
clear;
[trainImages,trainLabels, validatimages, validatLabels] = loadMNIST('mnist/train-images.idx3-ubyte', 'mnist/train-labels.idx1-ubyte','mnist/t10k-images.idx3-ubyte','mnist/t10k-labels.idx1-ubyte');

load('resources/trainedToyNet_v01');    % Load pretrained ToyNet

dYdX = computedYdX(tn,1);

% Pick image then classify image and print result in the console.
index = 33;     % Pick some image by its index
testImg =  validatimages(:,index);
[~,digitNumber] = max(validatLabels(:,index))
perturbedImg = testImg;
classifRes = ones(10,1);
eta = 0.02;

out = f(testImg,0,tn);
[~,l] = max(out);

adv = mod_adversarial_perturbation(testImg,l,@Df,@f,tn,eta);

figure(1)
subplot(1,2,1)
digitPerturbed = reshape(testImg+adv.r, [28,28]);    % row = 28 x 28 image
imshow(digitPerturbed*255, [0 255])      % show the image
title('perturbed');
subplot(1,2,2);
digitClean = reshape(testImg, [28,28]);    % row = 28 x 28 image
imshow(digitClean*255, [0 255]);      % show the image
title('original');



function out = f(testImg,flag,tn)
    out = forwardProp(tn,testImg)'; %do forward pass

    %flag==0:compute the outputs
    %flag==1:compute the label
    if flag==1
        [~,out] = max(out);
    end
end


function dzdx = Df(testImg,testLabel,idx,eta,tn)

    forwardProp(tn,testImg); %do forward pass

    for i=1:numel(idx)
        % dzdy = zeros(net.blobs(net.blob_names{end}).shape,'single');

        % dzdy(idx(i)) = 1
        backProp(tn,testImg,testLabel,eta, false);
        res = computedYdX(tn,idx(i));
        % res = net.backward({dzdy}); %do backward pass   % Get gradients with respect to params (vec should be of the size of the input image)
        % Make a vertical vector of the size of an image vector then stack it in a matrix
        dzdx(:,i) = reshape(res,numel(testImg),1);
    end
    dzdx = dzdx-repmat(dzdx(:,idx==testLabel),1,numel(idx));
end
