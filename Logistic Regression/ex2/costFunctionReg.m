function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X * theta;
h = sigmoid(z);

% The regularization term is the squared sum of all theta values, excluding theta(1)
reg = lambda / (2*m) * (sum(theta .^ 2)- theta(1)^2);

J = (1 / m) * sum(-y .* log(h) - (1-y) .* log(1 - h)) + reg;

tempTheta = zeros(size(X, 2), 1);

reg_dev = 0
for j = 1:size(X, 2)
	if (j > 1)
		reg_dev = (lambda / m) * theta(j);
	end
	
    tempTheta(j) = (1 / m) * sum((h - y) .* X(:,j)) + reg_dev;
end


grad  =  tempTheta;




% =============================================================

end
