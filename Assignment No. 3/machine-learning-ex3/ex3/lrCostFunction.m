function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% Initalize
J = 0;
grad = zeros(size(theta));

h=sigmoid(X*theta);

thetawoZero=[0; theta(2:size(theta,1))]; % Set the first theta value to zero, since it is the bias term and we dont regularize it
J=(1/m)*sum(-y.*log(h)-(1-y).*log(1-h))+(lambda/(2*m))*sum(thetawoZero.^2); %Regularized cost
grad=(1/m).*(X'*(h-y))+(lambda/m)*thetawoZero; %Regularized gradient

grad = grad(:);

end
