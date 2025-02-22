function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
min_error = inf;
for c_try = vals     
	for s_try = vals
	   model= svmTrain(X, y, c_try, @(x1, x2) gaussianKernel(x1, x2, s_try)); 
	   predictions = svmPredict(model, Xval);
	   curr_error = mean(double(predictions ~= yval));
	   %fprintf('Trying: c_try = %f, s_try = %f, error = %f \n---', c_try, s_try, curr_error );	    
	   if( curr_error < min_error )
	       min_error = curr_error;
	       C = c_try;
		   sigma = s_try;
		   fprintf('Updated: C = %f, sigma = %f, error = %f \n---', C, sigma, min_error);
	   end   
	end	
end

printf('Optimal C: ');disp(C);
printf('Optimal sigma: ');disp(sigma);

% =========================================================================

end
