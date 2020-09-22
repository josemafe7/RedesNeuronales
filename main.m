%Trabajo Redes Neuronales

%Autor: José_Manuel_Fernández_Labrador

%% Initialization
clear ; close all; clc


%% =========== Paso 1: Cargar y visualizar datos =============
fprintf('\nPaso 1: Cargar y visualizar datos\n')

%  We start the exercise by first loading the dataset. 

% Load Training Data
fprintf('Loading Data ...\n')

data = load('data_15.txt');

X = data(:,1:2);
y = data(:,3);

plotData(X,y);

m = size(X, 1);

X_train=X;
y_train=y;


%% ================ Paso 2. Red neuronal con una capa oculta de 2 neuronas y predicción ================
fprintf('\nPaso 2. Red neuronal con una capa oculta de 2 neuronas y predicción\n')
%% Setup the parameters you will use for this exercise
input_layer_size  = 2;
hidden_layer_size = 2;

fprintf('\nInitializing Neural Network Parameters ...\n')


initial_Theta1=[-0.0893, -0.0789, 0.0147; 0.1198, -0.1122, 0.0916];
initial_Theta2=[0.0406, -0.0743, -0.0315];

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


% Verificando J y calculo de derivadas

 [J grad] = nnCostFunction(initial_nn_params,input_layer_size,hidden_layer_size,X, y);
                                
 fprintf('Cost at initial theta: %f\n', J);
 fprintf('Gradient at initial theta: \n');
 fprintf(' %f \n', grad);
 
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will use "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 1000,'GradObj','on');


% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   X_train, y_train);

[nn_params, cost] = fminunc(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 1, (hidden_layer_size + 1));

Theta1
Theta2

plot_decision_boundary(Theta1,Theta2, X, y);
title('Numero neuronas 2');


%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the X set.

pred_train = predict(Theta1, Theta2, X_train);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred_train == y_train)) * 100);


%% ================ Paso 3. Red neuronal con una capa oculta de 1, 2, 3, 4, 5, 10, 20 y 50 neuronas y predicción ================
%% Setup the parameters you will use for this exercise
fprintf('\nPaso 3. Red neuronal con una capa oculta de 1, 2, 3, 4, 5, 10, 20 y 50 neuronas y predicción\n')
X = data(:,1:2);
y = data(:,3);
m = size(X, 1);

X_train=X;
y_train=y;

input_layer_size  = 2;
hidden_layer_size = 50;

fprintf('\nInitializing Neural Network Parameters ...\n')


initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, 1);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


% Verificando J y calculo de derivadas

 [J grad] = nnCostFunction(initial_nn_params,input_layer_size,hidden_layer_size,X, y);
                                
 fprintf('Cost at initial theta: %f\n', J);
 fprintf('Gradient at initial theta: \n');
 fprintf(' %f \n', grad);
 
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will use "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 1000,'GradObj','on');


% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   X_train, y_train);

[nn_params, cost] = fminunc(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 1, (hidden_layer_size + 1));

Theta1
Theta2

plot_decision_boundary(Theta1,Theta2, X, y);
title('Numero neuronas 50');


%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the X set.

pred_train = predict(Theta1, Theta2, X_train);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred_train == y_train)) * 100);


%% ================ Paso 4. Red neuronal con una capa oculta de 10 neuronas sin overfitting y predicción ================
fprintf('\nPaso 4. Red neuronal con una capa oculta de 10 neuronas sin overfitting y predicción\n')
%% Setup the parameters you will use for this exercise
input_layer_size  = 2;
hidden_layer_size = 10;
X = data(:,1:2);
y = data(:,3);
m = size(X, 1);

% Para tarbajar con todo el conjunto de datos
Xtrain=X;
ytrain=y;

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, 1);

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('\nTraining Neural Network... \n')

%  Initializing the maximun number of iterations
options = optimset('MaxIter', 1000, 'GradObj','on');

% Initializing the lambda regularization parameter
lambda = 1;

    
    % Create the cost function to be minimized
    costFunction = @(p) nnCostFunction_reg(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   Xtrain, ytrain,lambda);

    % Optimize the cost function by means of fmincg
    [nn_params, cost] = fminunc(costFunction, initial_nn_params, options);

    % Obtain Theta1 and Theta2 back from nn_params
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 1, (hidden_layer_size + 1));

    plot_decision_boundary(Theta1,Theta2, X, y);
    title('Lambda 1');
    fprintf('Program paused. Press enter to continue.\n');
    pause;
    
    pred = predict(Theta1, Theta2, X);
    
    % Compute the accuracy of the X set
    acierto = mean(double(pred == y)) * 100;
    fprintf('\nFull Set Accuracy with lambda %f: %f\n', lambda,acierto);

