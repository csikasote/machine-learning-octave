clear ; close all; clc

% Loading and visualizing data
fprintf('Loading and visualizing data ...\n')


% Developing a dataset D
A = ones(1,100).*linspace(0,1,100);
B = sin(2*pi*A) + cos(4*pi*A) + rand(1,100)*0.2;
D = [A;B]'; % Dataset

% Partitioning the dataset into training 50 and validation 50
[m,n] = size(D) ;
P = 0.50 ;
idx = randperm(m)  ;
train = D(idx(1:round(P*m)),:) ; 
val = D(idx(round(P*m)+1:end),:);

X = train(:,1); 
y = train(:,2);

Xval = val(:,1);
yval = val(:,2);


figure(1)
plot(X,y,'o')
title(sprintf('Visualizing training data'));
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
legend ('Training')
axis ([0, 1, -2, 1.5]);


fprintf('Program paused. Press enter to run the Neural Network.\n');
pause;
  
lambda = 0.001;
options = optimset('MaxIter', 500);


% Setting the Network Architecture
[n_x, d_x, n_h, n_y] = nn_layer_sizes(X,y);

% Randomly initializing the parameters (Theta1 and Theta2
initial_Theta1 = randInitializeWeights(d_x, n_h);
initial_Theta2 = randInitializeWeights(n_h, n_y);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% Create "short hand" for the cost function to be minimized
[nn_params, cost] = ...
    nnTrainLinearReg(initial_nn_params,d_x,n_h,n_y,X, y, lambda);

J = nnCostFunctionLinear(nn_params, d_x, n_h,n_y, X, y, lambda);

fprintf(['Cost computed by a NN for linear system: %f '...
         '\n\n'], J);

fprintf('Program paused. Press enter to continue.\n');
pause;

figure(2)
iters = length(cost);
plot(1:iters, cost);
title(sprintf('Learning Curve for NN (lambda = %d)', lambda));
xlabel('Number of Iterations');
ylabel('Errors');


% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:n_h * (d_x + 1)), ...
                 n_h, (d_x + 1));

Theta2 = reshape(nn_params((1 + (n_h * (d_x + 1))):end), ...
                 n_y, (n_h + 1));

figure(3)
x = (min(X) - 15: 0.05 : max(X) + 25)';
y_pred = predictLinear(Theta1, Theta2, x);
plot(X, y, 'o');
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
hold on;
plot(x, y_pred, '--')
legend ('Training', 'NNLinear')
axis ([0, 1, -2, 1.5]);
hold off;

fprintf('Program paused. Press enter to continue.\n');
pause;

figure(4);
[error_train, error_val] = ...
    nnLearningCurve(initial_nn_params, d_x, n_h, ...
                   n_y,X, y, Xval, yval, lambda);
plot(1:size(X,1), error_train);
hold on;
plot(1:size(Xval,1), error_val);
title(sprintf('Learning Curve for NN (lambda = %d)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 50 0 0.5])
legend('Train', 'Cross Validation')

fprintf('Neural Network for Linear Systems (lambda = %d)\n\n', lambda);
fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:size(X,1)
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

[lambda_vec, error_train, error_val] = ...
    nnValidationCurve(initial_nn_params,d_x,n_h,n_y,X, y, Xval, yval);
    
#close all;
figure(5);
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');
axis([0 0.1 0 0.5])

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;