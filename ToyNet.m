classdef ToyNet
    properties
        numHiddenLayers;
        inputLayerSize;
        outputLayerSize;
        hiddenLayersSize;
        arrayWeights;
        arrayBiases;
        Y;      % An array Y stores activation layaer vectors
        D;      % An arrat that stores Delta vectors. Delta represent the derivative of of the CostFunction w.r.t. z vector (z = Wa+b)
    end

    methods
        % ToyNet constructor
        function obj = ToyNet(i_numHiddenLayers, i_inputLayerSize, i_outputLayerSize, i_hiddenLayersSize)
            obj.numHiddenLayers = i_numHiddenLayers;
            obj.inputLayerSize = i_numHiddenLayers;
            obj.outputLayerSize = i_outputLayerSize;
            obj.hiddenLayersSize = i_hiddenLayersSize;
            obj.totalNumLayers = i_numHiddenLayers + 2;
            rng(5000); % seed generator

            % Init arrays
            obj.D{1} = 0;
            obj.Y{1} = 0;
            obj.Y{2} = zeros(obj.hiddenLayersSize, 1);    % init array Y as matrix with all enries 0
            obj.D{2} = zeros(obj.hiddenLayersSize, 1);    % init array D as matrix with all enries 0

            % Build W2, b2 for connections from input layer to first hidden
            W2 = 0.5*rand(obj.hiddenLayersSize, obj.inputLayerSize);
            b2 = 0.5*rand(obj.hiddenLayersSize,1);

            obj.arrayWeights{2} = W2;
            obj.arrayBiases{2} = b2;

            % Build intermediate W and b
            for i = 3:obj.totalNumLayers - 1   % do not build W and b from last hidden layer to output layer
                W = 0.5*rand(obj.hiddenLayersSize, obj.hiddenLayersSize);
                b = 0.5*rand(obj.hiddenLayersSize,1);
                obj.arrayWeights{i} = W;
                obj.arrayBiases{i} = b;
            end

            % Build W and b from last hidden layer to output layer
            WN = 0.5*rand(obj.outputLayerSize, obj.hiddenLayersSize);
            bN = 0.5*rand(obj.outputLayerSize, 1);
            obj.arrayWeights{i + 1} = WN;
            obj.arrayBiases{i + 1} = bN;

        end


        % Forward propagation
        function result = forwardProp(i_vector)
            % activate first hidden layer
            obj.Y{2} = activate(i_vector, obj.arrayWeights{2}, obj.arrayBiases{2});

            % activate other consequent layers plus output layer
            for i = 3:obj.totalNumLayers
                obj.Y{i} = activate(a, obj.arrayWeights{i}, obj.arrayBiases{i});
            end

            result = a;

        end


        % Back propagation
        function backresult = backprop(label_vector)
            % Backward pass
            YN = obj.totalNumLayers;
            obj.D{YN} = obj.Y{YN} .* (1 - obj.Y{YN}) .* (obj.Y{YN} - label_vector);

            for i = YN-1:-1:2
                obj.D{i} = obj.Y{i} .* (1 - obj.Y{i}) .* (obj.arrayWeights{i+1}' * obj.D{i+1});
            end

            % Gradient step
            obj.arrayWeights{2} = obj.arrayWeights{2} - eta * obj.D{2} * i_vector';

            for i = 3:YN
                obj.arrayWeights{i} = obj.arrayWeights{i} - eta * obj.D{i} * obj.Y{i - 1}
            end

            % delta4 = a4.*(1-a4).*(a4-y(:,k));    % derivSigmra*derivCostWRTa4
            %
            % delta3 = a3.*(1-a3).*(W4'*delta4);
            % delta2 = a2.*(1-a2).*(W3'*delta3);
            %
            % % Gradient step
            % W2 = W2 - eta*delta2*x';    % weight - deriv of a weight
            % W3 = W3 - eta*delta3*a2';
            % W4 = W4 - eta*delta4*a3';
        end



        % Activation function
        function y = activate(x,W,b)
            %ACTIVATE Evaluates sigmoid function.
            %
            % x is the input vector, y is the output vector
            % W contains the weights, b contains the shifts
            %
            % The ith component of y is activate((Wx+b)_i)
            % where activate(z) = 1/(1+exp(-z))
                y = 1./(1+exp(-(W*x+b)));
        end
    end
end
