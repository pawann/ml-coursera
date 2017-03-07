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


%printf('Size of Theta1: ');
%disp(size(Theta1));

%printf('Size of Theta2: ');
%disp(size(Theta2));

%printf( 'Y 5 sample values and zie:');
%disp(y(1:5));
%disp( size(y) );

%pause;

X = [ones(size(X,1), 1), X];
%for each training sample
for i=1:m

  y_val = y(i);
  
  % Forward propogation  
  A1 = X(i, :);
  Z2 = Theta1 * A1';
  
  A2 = sigmoid(Z2);
  A2 = [ 1; A2 ];
  
  % A3 is H  
  Z3 = Theta2 * A2;
  H_k = sigmoid(Z3);
  
  %transform number to Y_k vector
  Y_k = ( [ 1 : num_labels ] == y_val )';
  
  val = (-1/m) * sum( Y_k .* log( H_k ) + (1 - Y_k) .* log(1- H_k) );
   
  J = J + val;   
  
  %printf('i: %d, y: %d, val: %f, J:%f\n', i, y_val, val, J); 
  
  %% %% %% %% 
  %% Back propogation
  %% %% %% %% 
  
  % Error in output (3rd) layer
  D3 = H_k - Y_k;
  
  % Error in Hidden layer l2
  %printf('Size of Theta2-dash: '); disp( size(Theta2') );
  %printf('Size of D3: '); disp( size(D3) );
  Z2 = [1 ; Z2];
  %printf('Size of Z2: '); disp( size(Z2) );
  
  D2 = (Theta2' * D3) .* sigmoidGradient(Z2);
  D2 = D2(2:end);

  %printf('Size of DELTA_2: '); disp( size(DELTA_2) );
  %printf('Size of Theta1_grad: '); disp( size(Theta1_grad) );
  %printf('Size of Theta2_grad: '); disp( size(Theta2_grad) );
  %printf('Size of D2: '); disp( size(D2) );
  %printf('Size of A1: '); disp( size(A1) );
  %printf('Size of D3: '); disp( size(D3) );
  %printf('Size of A2: '); disp( size(A2) );

  % Accumulate the gradient from current training example    
  Theta2_grad = Theta2_grad + D3 * A2';
  Theta1_grad = Theta1_grad + D2 * A1;
  
end

  Theta2_grad = 1/m * Theta2_grad;
  Theta1_grad = 1/m * Theta1_grad;


%Ignore bias values in first column of thetas 
lambda_term = (lambda/(2*m)) * ( sum(sum(Theta1(:, 2:size(Theta1,2)) .^2)) + sum(sum(Theta2(:, 2:size(Theta2,2)) .^2)) );

J = J + lambda_term;

%Regularize theta gradients

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m) * Theta2(:, 2:end);

%printf('Final J = %f\n', J);

%pause;












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
