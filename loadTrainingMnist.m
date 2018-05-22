images = loadMNISTImages('mnist/train-images.idx3-ubyte');  
labels = loadMNISTLabels('mnist/train-labels.idx1-ubyte');
labels = labels';
labels(labels==0) = 10;    % replace all 0 with 10
labels = dummyvar(labels);      % make a matrix 60000x10


figure                                          % initialize figure
colormap(gray)                                  % set to grayscale
for i = 1:36                                    % preview first 36 samples
    subplot(6,6,i)                              % plot them in 6 x 6 grid
    digit = reshape(images(:, i), [28,28]);     % row = 28 x 28 image
    imagesc(digit)                              % show the image
    title(num2str(labels(i)))                   % show the label
end


