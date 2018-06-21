function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


% Using R To compute the unregularized cost. R that are 
% zero will remove the contribution of corresponding movies, which user 
% has not rated.

unreg_cost = 0.5 * sum(sum(((X * Theta' - Y) .^ 2) .* R))

% The regularized cost terms for X and Theta 

reg_cost_X = (lambda / 2) * sum(sum(X .^ 2));
reg_cost_Theta = (lambda / 2) * sum(sum(Theta .^ 2));

% The cost function is sum of these three terms

J = unreg_cost + reg_cost_X + reg_cost_Theta;

% Note, tutorial link:
% https://www.coursera.org/learn/machine-learning/module/HjnB4/discussions/92NKXCLBEeWM2iIAC0KUpw


% compute the error matrix by using R matrix to filter out all unrated movies

error = X * Theta' - Y;
error_factor = error .* R;

% X gradient can be calculated by using a matrix
% multiplication between error_factor matrix and Theta. 
% regularization: lambda multiplied by X.

X_grad = (error_factor * Theta) + (lambda * X);

% Do something similar for the Theta gradients, but need to transpose
% the error_factor matrix. regularization: lambda multiplied by Theta.
%
Theta_grad = (error_factor' * X) + (lambda * Theta);





% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
