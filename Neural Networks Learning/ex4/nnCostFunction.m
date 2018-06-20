function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1:
%% forward propagation %%
% add bias for input layer
X = [ones(m, 1) X]; % 5000*401

% hidden layer calculation
a1 = X;
z2 = Theta1 * a1'; % (25*401)*(401*5000)
a2 = sigmoid(z2); % (25*5000)

% add bias for hidden layer
a2 = [ones(1, size(a2, 2)); a2]; % (26*5000)

% output layer calculation
z3 = Theta2 * a2; % (10*26) *(26*5000)
a3 = sigmoid(z3); % (10*5000)
h = a3;

% resize y as a label vector as one hot coe for all sample (only values 0 or 1) (e.g. 1 -> [0;1;0;0;0;0;0;0;0;0])
y_coded = zeros(num_labels, m); % (10*5000)
for i=1:m,
  y_coded(y(i),i)=1;
end

%% Calculation of the cost without regularization
J = (1/m) * sum ( sum ( (-y_coded) .* log(h) - (1-y_coded) .* log(1-h) ));

% Do not regularize the terms that correspond to the bias. They are the first column of each Theta matrix.
t1 = Theta1(:,2:size(Theta1,2));
t2 = Theta2(:,2:size(Theta2,2));

% The regularization term is the squared sum of all theta values, excluding bias
reg = lambda / (2*m) * (sum(sum(t1 .^ 2)) + sum(sum(t2 .^ 2)));

% Regularized cost function
J = J + reg;



%Part2
% Back propagation
for t=1:m

    % forward propagation
	a1 = X(t,:); 		% already has bias (1*401)
	z2 = Theta1 * a1'; 	% (25*401)*(401*1)
	a2 = sigmoid(z2); 	% (25*1)
    
    a2 = [1 ; a2]; 		% adding bias (26*1)
	z3 = Theta2 * a2; 	% (10*26)*(26*1)
	a3 = sigmoid(z3); 	% (10*1)
    
    % Calculate error and delta
	delta_3 = a3 - y_coded(:,t); % (10*1)
	
    z2=[1; z2]; % add bias (26*1)

    delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z2); % ((26*10)*(10*1))=(26*1)
	delta_2 = delta_2(2:end); % skipping delta_2(0) (25*1)

	Theta2_grad = Theta2_grad + delta_3 * a2'; % (10*1)*(1*26)
	Theta1_grad = Theta1_grad + delta_2 * a1; % (25*1)*(1*401)    
end;

% calculate gradient
Theta2_grad = (1/m) * Theta2_grad; % (10*26)
Theta1_grad = (1/m) * Theta1_grad; % (25*401)



%Part3
% Regularization

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end)); % for j >= 1 
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end)); % for j >= 1



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
