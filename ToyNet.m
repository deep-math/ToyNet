% This neural network is based on the publication by Catherine F. Higham and Desmond J. Higham

classdef ToyNet < handle
    properties
        numHiddenLayers;
        inputLayerSize;
        outputLayerSize;
        hiddenLayersSize;
        arrayWeights;
        arrayBiases;
        totalNumLayers;
        Y;      % An array Y stores  layaer vectors
        D;      % An arrat that stores Delta vectors. Delta represent the derivative of of the CostFunction w.r.t. z vector (z = Wa+b)
    end

    methods
        % ToyNet constructor
        function obj = ToyNet(i_numHiddenLayers, i_inputLayerSize, i_outputLayerSize, i_hiddenLayersSize)
            obj.numHiddenLayers = i_numHiddenLayers;
            obj.inputLayerSize = i_inputLayerSize;
            obj.outputLayerSize = i_outputLayerSize;
            obj.hiddenLayersSize = i_hiddenLayersSize;
            obj.totalNumLayers = i_numHiddenLayers + 2;
%             rng(5000); % seed generator

            % Init arrays
            obj.D{1} = 0;
            obj.Y{1} = 0;
            obj.Y{2} = zeros(obj.hiddenLayersSize, 1);    % init array Y as matrix with all enries 0
            obj.D{2} = zeros(obj.hiddenLayersSize, 1);    % init array D as matrix with all enries 0

            % Build W2, b2 for connections from input layer to first hidden
            W2 = 0.5*ones(obj.hiddenLayersSize, obj.inputLayerSize);
            b2 = 0.5*ones(obj.hiddenLayersSize,1);

            obj.arrayWeights{2} = W2;
            obj.arrayBiases{2} = b2;


            % Build intermediate W and b
            for i = 3:obj.totalNumLayers - 1   % do not build W and b from last hidden layer to output layer
                W = 0.5*ones(obj.hiddenLayersSize, obj.hiddenLayersSize);
                b = 0.5*ones(obj.hiddenLayersSize,1);
                obj.arrayWeights{i} = W;
                obj.arrayBiases{i} = b;
            end

            % Build W and b from last hidden layer to output layer
            WN = 0.5*ones(obj.outputLayerSize, obj.hiddenLayersSize);
            bN = 0.5*ones(obj.outputLayerSize, 1);
            obj.arrayWeights{obj.totalNumLayers} = WN;
            obj.arrayBiases{obj.totalNumLayers} = bN;

        end


        % Forward propagation
        function result = forwardProp(obj, i_vector)

            % activFunc first hidden layer
            obj.Y{2} = activFunc(i_vector, obj.arrayWeights{2}, obj.arrayBiases{2});

            % activFunc other consequent layers plus output layer
            for i = 3:obj.totalNumLayers
                obj.Y{i} = activFunc(obj.Y{i-1}, obj.arrayWeights{i}, obj.arrayBiases{i});
            end

            result = obj.Y;

        end


        % Back propagation
        function backresult = backProp(obj, i_vector, label_vector, eta)
            % Backward pass
            YN = obj.totalNumLayers;

            % Calculate the last layer error gradient dC/dZ
            obj.D{YN} = obj.Y{YN} .* (1 - obj.Y{YN}) .* (obj.Y{YN} - label_vector);

            % Calculate error gradient for L-1, L-2,..., 2 layers
            for i = YN-1:-1:2
                obj.D{i} = obj.Y{i} .* (1 - obj.Y{i}) .* (obj.arrayWeights{i+1}' * obj.D{i+1});
            end

            % disp({'NN delta2' obj.D{YN} 'NN delta1' obj.D{YN-1}});

            % Gradient step. Update weights and biases
            obj.arrayWeights{2} = obj.arrayWeights{2} - eta * obj.D{2} * i_vector';
            obj.arrayBiases{2} = obj.arrayBiases{2} - eta* obj.D{2};

            for i = 3:YN
                obj.arrayWeights{i} = obj.arrayWeights{i} - eta * obj.D{i} * obj.Y{i - 1}';
                obj.arrayBiases{i} = obj.arrayBiases{i} - eta *obj.D{i};
            end

            disp({'NN W2' obj.arrayWeights{2} 'NN b2' obj.arrayBiases{2}});
            disp({'NN W3' obj.arrayWeights{3} 'NN b3' obj.arrayBiases{3}});
        end

        % Training
        function trainingRes = train(obj, trainData, trainLabel, cycles, eta)
            [vecSize, numVecs] = size(trainData);

            for i = 1:cycles
                randInd = randi(numVecs);
                x = trainData(:, randInd);
                y = trainLabel(:, randInd);
                forwardProp(obj, x);
                backProp(obj, x, y, eta);
            end

        end

        % Classify the input x
        function cc = classify(obj, x)
            listLayers = forwardProp(obj, x);
            [numRows, numCols] = size(listLayers);
            cc = listLayers(numCols);
        end

        % function costResult = cost(obj, x, y)
        %     layersList = forwardProp(obj, x);
        %     [listRows, listCols] = size(layersList);
        %     lastActivLayer = layersList[listCols];
        %     costvec(i) = norm(y(:,i) - a4, 2);
        % end


    end
end


%  activation function
function y = activFunc(x, W, b)
    %activFunc Evaluates sigmoid function.
    %
    % x is the input vector, y is the output vector
    % W contains the weights, b contains the shifts
    %
    % The ith component of y is activFunc((Wx+b)_i)
    % where activFunc(z) = 1/(1+exp(-z))
    y = 1./(1+exp(-(W*x+b)));
end
