clc
close all
clear all

%% basic information about the problem
myFN = @(X) (6*X - 2).^2 .* sin(12*X - 4);  % this could be any user-defined function
designspace = [0, 1];

% create DOE
npoints = 8;
% X = linspace(designspace(1), designspace(2), npoints)';
X = [0,0.15,0.25,0.31,0.33,0.37,0.5,1]';

% evaluate analysis function at X points
Y = feval(myFN, X);

% fit surrogate models
options = srgtsKRGSetOptions(X, Y);

[surrogate state] = srgtsKRGFit(options);

% create test points
Xtest = linspace(designspace(1), designspace(2), 100)';


% evaluate surrogate at Xtest
PredVar = srgtsKRGPredictionVariance(Xtest, surrogate);

plot(X, zeros(npoints, 1), 'o', ...
     Xtest, PredVar)