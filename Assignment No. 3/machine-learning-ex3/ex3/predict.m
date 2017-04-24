function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

temp1=[ones(m,1) X]; %Layer 1-2
temp2=[ones(m, 1) sigmoid(temp1*Theta1')]; %Layer 2-3
temp3=sigmoid(temp2*Theta2'); %Layer 2-output
[maxTemp3, maxTemp3_Idx]=max(temp3');
p=maxTemp3_Idx'; %Ouput P is the array that 


end
