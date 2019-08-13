function [n_x, d_x, n_h, n_y] = nn_layer_sizes(X,Y)
  
  % Arguments:
  % X -- input dataset of dimension (number of examples, input size)
  % Y -- labels of shape (output size, number of examples)
  
  % Returns:
  % n_x -- the size of the input layer
  % d_x -- dimension of the input layer
  % n_h -- the size of the hidden layer
  % n_y -- the size of the output layer
  
  n_x = size(X,1);
  d_x = size(X,2);
  n_h = 10;
  n_y = size(Y,2);
  
  end 