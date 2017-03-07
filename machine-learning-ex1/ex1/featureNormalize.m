function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

	for featureNum = 1 : size(X, 2)
       stdDev = std( X(:, featureNum) );
       meanVal = mean( X(:, featureNum) );	
       mu(featureNum) = meanVal;
	   sigma(featureNum) = stdDev;
	   
       for rowNum = 1 : size(X, 1)
	      featureVal = X(rowNum, featureNum);
	      X_norm(rowNum, featureNum) = (featureVal - meanVal)/stdDev;
	   end;
   end;








% ============================================================

end
