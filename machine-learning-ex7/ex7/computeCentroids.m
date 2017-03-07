function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

for c = 1 : K
 
  % Sum of Xs that are assigned to centroid c, a nx1 vector
  SumXc = zeros(n, 1);
  
  % Count of Xs that are assigned to centroid c, a 1x1 number
  countXc = 0;
  
  for i = 1 : m      
      if( idx(i) == c )
        SumXc = SumXc + (X(i, :))';
		countXc = countXc + 1;
      end
  end
  
  if(countXc != 0)
    %disp(size(centroids(c, :)));
	%disp(size(SumXc'));
	%disp(countXc);
	%disp(size(X));
    centroids(c, :) = (1/countXc) * SumXc';
  end
  
end





% =============================================================


end

