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
        DY;     % An array of derivatives dY/Dz  , i.e derivative of output neuron w.r.t. z
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
            % rng(5000); % seed generator
            initScaler = 0.1;
            % Init arrays
            obj.D{1} = 0;
            obj.Y{1} = 0;
            obj.DY{1} = 0;
            obj.Y{2} = zeros(obj.hiddenLayersSize, 1);    % init array Y as matrix with all enries 0
            obj.D{2} = zeros(obj.hiddenLayersSize, 1);    % init array D as matrix with all enries 0

            % Build W2, b2 for connections from input layer to first hidden
            W2 = initScaler*rand(obj.hiddenLayersSize, obj.inputLayerSize);
            b2 = initScaler*rand(obj.hiddenLayersSize,1);

            obj.arrayWeights{2} = W2;
            obj.arrayBiases{2} = b2;


            % Build intermediate W and b
            for i = 3:obj.totalNumLayers - 1   % do not build W and b from last hidden layer to output layer
                W = initScaler*rand(obj.hiddenLayersSize, obj.hiddenLayersSize);
                b = initScaler*rand(obj.hiddenLayersSize,1);
                obj.arrayWeights{i} = W;
                obj.arrayBiases{i} = b;
            end

            % Build W and b from last hidden layer to output layer
            WN = initScaler*rand(obj.outputLayerSize, obj.hiddenLayersSize);
            bN = initScaler*rand(obj.outputLayerSize, 1);
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

            result = obj.Y{end};

        end


        % Back propagation. Inputs: self, training vector, label vector, learning rate, id for dY/dZ where Y some neuron in the last layer
        function backresult = backProp(obj, i_vector, label_vector, eta, updateWeights)
            % Backward pass
            YN = obj.totalNumLayers;

            % Calculate the last layer error gradient dC/dZ
            obj.D{YN} = obj.Y{YN} .* (1 - obj.Y{YN}) .* (obj.Y{YN} - label_vector);

            % Calculate error gradient for L-1, L-2,..., 2 layers
            for i = YN-1:-1:2
                obj.D{i} = obj.Y{i} .* (1 - obj.Y{i}) .* (obj.arrayWeights{i+1}' * obj.D{i+1});
            end

            backresult = obj.D{2};

            if updateWeights == true
                % Gradient step. Update weights and biases
                obj.arrayWeights{2} = obj.arrayWeights{2} - eta * obj.D{2} * i_vector';
                obj.arrayBiases{2} = obj.arrayBiases{2} - eta* obj.D{2};

                for i = 3:YN
                    obj.arrayWeights{i} = obj.arrayWeights{i} - eta * obj.D{i} * obj.Y{i - 1}';
                    obj.arrayBiases{i} = obj.arrayBiases{i} - eta *obj.D{i};
                end
            end
        end


        %  Compute derivate dY_i/dx
        function dYdx = computedYdX(obj, Yid)
            dYdZ = 0;
            YN = obj.totalNumLayers;

            % Calculate the last layer gradient dY/dZ
            obj.DY{YN} = obj.Y{YN}(Yid)*(1-obj.Y{YN}(Yid));

            % Calculate the L-1 layer gradient dY_i/dZ
            obj.DY{YN-1} = obj.DY{YN} * obj.arrayWeights{YN}(Yid,:)' .* obj.Y{YN-1} .* (1 - obj.Y{YN-1});

            for i = YN-2:-1:2
                % Calculate error for dY/dZ
                dYdZ = obj.Y{i} .* (1 - obj.Y{i}) .* (obj.arrayWeights{i+1}' * obj.DY{i+1});
            end
            dYdx = obj.arrayWeights{2}'*dYdZ;
        end


        % Adversarial back prop
        function perturbedVector = adversBackProp(obj, i_vector,label_vector,eta)
            % Gradient w.r.t the i_vector
            backProp(obj, i_vector, label_vector, eta, false);
            perturbator = eta*obj.arrayWeights{2}'*obj.D{2};
            perturbedVector = i_vector + perturbator;
        end


        % Training
        function trainingRes = train(obj, trainData, trainLabel, cycles, eta)
            [vecSize, numVecs] = size(trainData);

            for i = 1:cycles
                randInd = randi(numVecs);
                x = trainData(:, randInd);
                y = trainLabel(:, randInd);
                forwardProp(obj, x);
                backProp(obj, x, y, eta, true);
            end

        end

        function delta = getDelta(obj, id)
            delta = obj.D{id};
        end

        function weights = getWeights(obj, id)
            weights = obj.arrayWeights(id);
            weights = weights{1};
        end
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
