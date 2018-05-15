classdef ToyNet
    properties
        numHiddenLayers;
        numActivationLayers;
        inputLayerSize;
        outputLayerSize;
        hiddenLayersSize;
        listOfWeightMatr;
        listOfBiasVecs;
        Y;      % Matrix Y. Each column of the matrix Y represents activation layer, rows represent the number of neurons in the layer
        D;      % Matrix delta. Each column of the matirx D represent the derivative of of the CostFunction w.r.t. z vector (z = Wa+b)
    end

    methods
        % ToyNet constructor
        function obj = ToyNet(i_numHiddenLayers, i_inputLayerSize, i_outputLayerSize, i_hiddenLayersSize)
            obj.numHiddenLayers = i_numHiddenLayers;
            obj.inputLayerSize = i_numHiddenLayers;
            obj.outputLayerSize = i_outputLayerSize;
            obj.hiddenLayersSize = i_hiddenLayersSize;
            obj.numActivationLayers = i_numHiddenLayers + 1;
            rng(5000); % seed generator

            % Build a matrix s.t. columns
            obj.Y = zeros(obj.hiddenLayersSize, obj.numActivationLayers);    % init matrix Y as matrix with all enries 0
            obj.D = zeros(obj.hiddenLayersSize, obj.numActivationLayers);    % init matrix Y as matrix with all enries 0

            % Build W2, b2 for connections from input layer to first hidden
            W2 = 0.5*rand(obj.hiddenLayersSize, obj.inputLayerSize);
            b2 = 0.5*rand(obj.hiddenLayersSize,1);

            obj.listOfWeightMatr = W2;
            obj.listOfBiasVecs = b2;

            % Build intermediate W and b
            for i = 1:obj.numHiddenLayers - 1   % do not build W and b from last hidden layer to output layer
                W = 0.5*rand(obj.hiddenLayersSize, obj.hiddenLayersSize);
                b = 0.5*rand(obj.hiddenLayersSize,1);
                obj.listOfWeightMatr(:,:,i+1) = W;
                obj.listOfBiasVecs(:,i+1) = b;
            end

            % Build W and b from last hidden layer to output layer
            WN = 0.5*rand(obj.outputLayerSize, obj.hiddenLayersSize);
            bN = 0.5*rand(obj.outputLayerSize, 1);
            obj.listOfWeightMatr(:,:, obj.numHiddenLayers) = WN;
            obj.listOfBiasVecs(:, obj.numHiddenLayers) = bN;

        end


        % Forward propagation
        function result = forwardProp(i_vector)
            % activate first hidden layer
            obj.Y(:,1) = activate(i_vector, obj.listOfWeightMatr(1), obj.listOfBiasVecs(1));

            % activate other consequent layers plus output layer
            for i = 2:obj.numActivationLayers
                obj.Y(:,i) = activate(a, obj.listOfWeightMatr(:,:,i), obj.listOfBiasvecs);
            end

            result = a;

        end


        % Back propagation
        function backresult = backprop(i_vector)
            % Backward pass
            Yn = obj.numActivationLayers;
            obj.D(:,Yn) = obj.Y(:,Yn).*(1-obj.Y(:,Yn)).*(obj.Y(:,Yn)-i_vector);

            for i = Yn-1:-1:1
                obj.D(:,i) = obj.Y(:,i).*(1-obj.Y(:,i)).*(obj.listOfWeightMatr(i+1)' * obj.D(i+1));
            end

            % Gradient step
            obj.listOfWeightMatr(:,:,1) = obj.listOfWeightMatr(:,:,1) - eta * obj.D(1) * i_vector';

            for i = 2:Yn
                obj.listOfWeightMatr(:,:,i) = obj.listOfWeightMatr(:,:,i) - eta * obj.D(i) * obj.Y(i-1)
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
