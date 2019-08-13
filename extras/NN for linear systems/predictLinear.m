function p = predictLinear(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
#num_labels = 1 %size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

a1   = [ones(m,1) X];
z2   = a1 * Theta1';
tz2 = tanh(z2);
a2   = [ones(m,1) tz2];
p   = a2 * Theta2';

#h1 = tanh([ones(m, 1) X] * Theta1');
#p2 = [ones(m, 1) h1] * Theta2';
#p = h2;
#[dummy, p] = max(h2, [], 2);
% =========================================================================

end
