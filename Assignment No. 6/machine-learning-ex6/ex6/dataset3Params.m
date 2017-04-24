function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

Cvals = [ 0.01 0.03 0.1 0.3 1 3 10 30 ];
mE = Inf;
mC = Inf;
mS = Inf;

for i = 1:length(Cvals)
    for j = 1:length(Cvals)
        currentC = Cvals(i);
        currentSigma = Cvals(j);
        model = svmTrain(X, y, currentC, @(x1, x2) gaussianKernel(x1, x2, currentSigma));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));

        if error < mE
            mE = error;
            mC = currentC;
            mS = currentSigma;
        end
    end
end

C = mC;
sigma = mS;

end
