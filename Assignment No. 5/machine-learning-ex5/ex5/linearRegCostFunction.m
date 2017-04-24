function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

h=X*theta; %Linear regression hypothesis
sq_err=(h-y).^2; %Square error
thetaNoZero=[0; theta(2:end)]; %Make new theta with 0 in 1 column
J=(1/(2*m))*(sum(sq_err))+(lambda/(2*m))*(sum(thetaNoZero.^2));
grad=(1/m).*(X'*(h-y))+(lambda/m)*thetaNoZero

grad = grad(:);

end
