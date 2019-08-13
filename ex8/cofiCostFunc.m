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

%total = sum(sum((A * B) .* R));
%C = (A * B) .* R; total = sum(C(:));

pred = X * Theta';
error = pred - Y;
error_factor = error .* R;
sqr_error = error_factor .^2;
total = sum(sqr_error(:));
reg_term1 = (lambda/2) * sum(X(:).^2);
reg_term2 = (lambda/2) * sum(Theta(:).^2);
unreg_cost = (1/2) * total;
J = unreg_cost + reg_term1 + reg_term2;

reg_x_grad = lambda * X;
reg_theta_grad = lambda * Theta;
X_grad = error_factor * Theta + reg_x_grad;
Theta_grad = error_factor' * X + reg_theta_grad;

% [m n] = size(X);
% for i=1:m, 
	% idx = find(R(i, :)==1);
	% Thetatemp = Theta(idx; :);
	% Ytemp = Y(i; idx);
	% X_grad (i, :) = (X(i, :) * Thetatemp' -Ytemp) * Thetatemp;
 % end

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
