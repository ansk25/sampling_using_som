clc
clear all
close all

%% Generate Data
%train the surrogate model
[X1, X2] = meshgrid(linspace(-1,1,15),linspace(1,-1,15));
X1(2:7,2:7) = 1.75*X1(2:7,2:7)+0.05;
X2(2:7,2:7) = 2.25*X2(2:7,2:7);

myFN = @(X) 1 + X1.^2 + X2.^2; 
% evaluate analysis function at X points
rows = size(X1,1)*size(X2,2);
x1 = [reshape(X1,[rows,1])];
x2 = [reshape(X2,[rows,1])];
%%
X_matrix = horzcat(x1, x2);
Y_matrix = feval(myFN, X_matrix);
rows = size(Y_matrix,1)*size(Y_matrix,2);
y_matrix = [reshape(Y_matrix,[rows,1])];
% fit surrogate models
options = srgtsKRGSetOptions(X_matrix, y_matrix);
%%
[surrogate state] = srgtsKRGFit(options);
%generate test points
% Define the range for each variable
range_X = [-1.70, 1.80];
range_Y = [-2.25, 2.25];

% Specify the number of points
numPoints = 25;

% Generate Latin Hypercube Samples
lhs_samples = lhsdesign(numPoints, 2);

% Map the LHS samples to the specified range
Xtest1 = lhs_samples(:, 1) * (range_X(2) - range_X(1)) + range_X(1);
Xtest2 = lhs_samples(:, 2) * (range_Y(2) - range_Y(1)) + range_Y(1);


% Map the LHS samples to the specified range
Z = 1 + Xtest1.^2 + Xtest2.^2; 
rows = size(Xtest1,1)*size(Xtest2,2);
xtest1 = [reshape(Xtest1,[rows,1])];
xtest2 = [reshape(Xtest2,[rows,1])];
rows = size(Z,1)*size(Z,2);
z = [reshape(Z,[rows,1])];

Xtest = horzcat(xtest1, xtest2);
%%
% evaluate surrogate at Xtest
PredVar = srgtsKRGPredictionVariance(Xtest, surrogate);

XY_matrix = Xtest;
disp(size(XY_matrix));
disp(size(Z));
% rows = size(x1,1)*size(x2,2);
data = [xtest1, xtest2, z];

%%
sData = som_data_struct(data,'my-data','comp_names',{'x1','x2','p-Matrix'});
sData = som_normalize(sData,'range');

%% Initializing SOM Map Codebook Vectors (Linear Initialization)

[sMap]= modifiedsom_lininit(sData,'lattice','hexa','msize',[5,5]);
sMap1 = sMap;

%% Training SOM
[sMap,sTrain] = modifiedsom_batchtrain(sMap,sData,'sample_order','ordered','trainlen',200);

%% Denormalizing the data
sMap=som_denormalize(sMap,sData); 
sData=som_denormalize(sData,'remove');

%% Density matrix
sMap_umatrix = sMap;
[p_mat, dist]= som_density_mat(sMap,data);

%% p-Matrix
% U = som_umat(sMap_umatrix);
% Um = U(1:2:size(U,1),1:2:size(U,2));
% Um = reshape(Um, [size(Um,1)*size(Um,2),1]);

sMap_umatrix_D = sMap_umatrix;
sMap_umatrix_D.codebook(:,1:2) = sMap_umatrix.codebook(:,1:2);
sMap_umatrix_D.codebook(:,3) = 1-p_mat;

%% Visualization of SOM results( U Matrix and Component Planes )
figure(2) 
som_show(sMap_umatrix_D,'comp','all');

%% 
% min_values = min(sMap_umatrix_D.codebook);  % Modify these values as needed
max_values = max(sMap_umatrix_D.codebook);  % Modify these values as needed
constraints = {'x1','x2','p-Matrix'};

%% Pareto Visualization
sliders = [0,0,1];
figure(3); hold on;
som_show_pi_ei(sMap_umatrix_D, constraints, 3, sliders,[], 'comp', 'all','bar','horiz');

%% iSOM Grid in function space  
figure(7)
som_grid(sMap_umatrix,'coord',sMap_umatrix_D.codebook,'label',sMap_umatrix.labels,'labelcolor','c','labelsize',5, 'marker','o','MarkerColor','k'...
    ,'MarkerSize',7,'linecolor', 'k');
hold on, scatter3(data(:,1),data(:,2),1-p_mat,20,'ro','filled');

xlabel('F1')
ylabel('F2')
zlabel('F3')

%% prediction variance

PredVar = som_normalize(PredVar,'range');
data = [reshape(Xtest1, [rows, 1]), reshape(Xtest2, [rows, 1]), PredVar];
sData = som_data_struct(data,'my-data','comp_names',{'x1', 'x2', 'predvar'});
sData = som_normalize(sData,'range');
% [sMap]= modifiedsom_lininit(sData,'lattice','hexa','msize',[20,20]);

[sMap1,sTrain] = modifiedsom_batchtrain(sMap1,sData,'sample_order','ordered','trainlen',200);
sMap1.comp_names= {'x1', 'x2', 'predvar'};

sMap1=som_denormalize(sMap1,sData); 
sData=som_denormalize(sData,'remove');

sMap_umatrix = sMap1;

sMap_umatrix_D = sMap_umatrix;
sMap_umatrix_D.codebook(:,1:2) = sMap_umatrix.codebook(:,1:2);

%% iSOM Grid in function space  
figure(8)
som_grid(sMap_umatrix,'coord',sMap_umatrix.codebook,'label',sMap_umatrix.labels,'labelcolor','c','labelsize',5, 'marker','o','MarkerColor','k'...
    ,'MarkerSize',7,'linecolor', 'k');
hold on, scatter3(data(:,1),data(:,2),data(:,3),20,'ro','filled');

xlabel('F1')
ylabel('F2')
zlabel('F3')

figure(10) 
som_show(sMap_umatrix_D,'comp','all');
% 
% function covariance_matrix = calculate_covariance(XY_matrix, Z)
%     % X and Y are matrices with the same number of observations
% 
%     % Calculate mean of X and Y
%     mean_XY = mean(XY_matrix);
%     mean_Z = mean(Z);
% 
%     % Subtract mean from X and Y
%     XY_centered = XY_matrix - mean_XY;
%     Z_centered = Z - mean_Z;
% 
%     % Calculate covariance matrix element-wise
%     covariance_matrix = (XY_centered * Z_centered') ./ (size(XY_matrix, 1) - 1);
% end