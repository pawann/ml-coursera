function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

% 1 row per dataset entry
sigma = zeros(size(X,2),1);
featureSize = size(X,2)
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

	
	
	H = X * theta;
	for featureNum = 1 : featureSize
	  x = X(:, featureNum);	  
	  sigma(featureNum) = sum((H - y) .* x);
	  %disp(sprintf('My loop..%d and sigma = %d, sum is %d', featureNum, sigma(featureNum), sum((H - y) .* x)));
	end;
	
	%sigma = ((theta' * X' - y')*X)';
	
	theta = theta - alpha/m * sigma;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
	%disp(sprintf('cost: %d',J_history(iter)));

end

end
