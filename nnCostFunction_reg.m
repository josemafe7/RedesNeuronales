function [J grad] = nnCostFunction_reg(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 1, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
n = size(X, 2);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

DELTA1= zeros(hidden_layer_size,input_layer_size+1);
DELTA2 = zeros(1,hidden_layer_size+1);

suma = 0;

for i=1:m
    a1 = X(i,:)';
    a1 = [1; a1];
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    a2 = [1; a2];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    delta3 = a3 - y(i);
    delta2 = (Theta2'* delta3) .* ((1-a2).*a2);
    delta2 = delta2(2:end);
    DELTA1 = DELTA1 + delta2 * a1';
    DELTA2 = DELTA2 + delta3 * a2';
    h = a3;  
    aux1 = log(h);
    aux2 = log(1 - h);
    suma = suma + y(i) * aux1 + (1-y(i)) * aux2;
end
J = (-1/m)*suma;
Theta1SIN = Theta1(:,2:end);
Theta2SIN = Theta2(:,2:end);
J = J+(lambda/(2*m))*(sum(sum(Theta1SIN.^2))+sum(sum(Theta2SIN.^2)));
		
								  
% -------------------------------------------------------------

Theta1_grad(:,1) = (1/m) * DELTA1(:,1);
Theta2_grad(:,1) = (1/m) * DELTA2(:,1);
Theta1_grad(:,2:end) = (1/m) * DELTA1(:,2:end) + (lambda/m) * Theta1SIN;
Theta2_grad(:,2:end) = (1/m) * DELTA2(:,2:end) + (lambda/m) * Theta2SIN;

% =========================================================================
Theta1_grad;
Theta2_grad;
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
