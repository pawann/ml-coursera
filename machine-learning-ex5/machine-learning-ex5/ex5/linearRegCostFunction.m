function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%printf('data set size m: '); disp(m);
%printf('Size of X: '); disp(size(X));
%printf('Size of Y: '); disp(size(y));
%printf('Size of theta: '); disp(size(theta));
%printf('Size of grad: '); disp(size(grad));
%printf('lambda: '); disp(lambda);

H = X * theta;
subtheta = theta(2:size(theta, 1), :);
J = (1/(2*m)) * sum(( H - y ).^ 2) + (lambda/(2*m)) * sum (  subtheta .^ 2);

%grad = (1/m) * sum ( (H - y ) .* X ) + (lambda/m) * [0; subtheta];

grad = (1/m) * ((theta' * X' - y')*X)' + (lambda/m) * [0; subtheta];

% =========================================================================

grad = grad(:);

end
