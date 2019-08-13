%% Initialization
clear ; close all; clc

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  The following code will load the dataset into your environment and plot
%  the data.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

% Load from ex5data1: 
% You will have X, y, Xval, yval, Xtest, ytest in your environment
load ('ex5data1.mat');

% m = Number of examples
m = size(X, 1);

% Plot training data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 6: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

%% Setup the parameters you will use for this exercise
input_layer_size  = 1;  
hidden_layer_size = 8;   
num_labels = 1;   
lambda = 0.01;
options = optimset('MaxIter', 500);

% Randomly initializing the parameters (Theta1 and Theta2
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% Normalizing the features
[X_norm, mu, sigma] = featureNormalize(X);
#[y_norm, mu_y, sigma_y] = featureNormalize(y);
[Xval_norm, mu_val, sigma_val] = featureNormalize(Xval);


% Create "short hand" for the cost function to be minimized
#costFunction = @(p) nnCostFunctionLinear(p,input_layer_size,hidden_layer_size, num_labels, X_norm, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
#[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
[nn_params, cost] = ...
    nnTrainLinearReg(initial_nn_params,input_layer_size,...
                      hidden_layer_size,num_labels,X_norm, y, lambda); 
J = nnCostFunctionLinear(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X_norm, y, lambda);

fprintf(['Cost computed by a NN for linear system: %f '...
         '\n\n'], J);
fprintf('Program paused. Press enter to continue.\n');
pause;

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

% ================================================================


figure(1)
x = (min(X) - 15: 0.05 : max(X) + 25)';
y_pred = predictLinear(Theta1, Theta2, x);
plot(X_norm, y, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
hold on;
plot(x, y_pred, 'o')
legend ('Training', 'NNLinear')
axis ([-2, 2, 0, 50]);
hold off;

figure(2);
lambda = 0.01;
[error_train, error_val] = ...
    nnLearningCurve(initial_nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels,X_norm, y, Xval_norm, yval, lambda);
plot(1:m, error_train, 1:m, error_val);

title(sprintf('Learning Curve for NN (lambda = %d)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')

fprintf('Neural Network for Linear Systems (lambda = %d)\n\n', lambda);
fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 8: Validation for Selecting Lambda =============
%  You will now implement validationCurve to test various values of 
%  lambda on a validation set. You will then use this to select the
%  "best" lambda value.
%

[lambda_vec, error_train, error_val] = ...
    nnValidationCurve(initial_nn_params,input_layer_size,hidden_layer_size,num_labels,X_norm, y, Xval_norm, yval);
    
#close all;
figure(3);
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;
