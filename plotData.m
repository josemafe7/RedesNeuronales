function plotData(x, y)
% ============================================================
%PLOTDATA Plots the data points x and y into a new figure 
% ============================================================

figure();

IP = (y == 1);%Indices positivos
IN = (y == 0);%Indices negativos
plot(x(IP,1), x(IP,2), 'bo', 'MarkerSize', 5);
hold on;
plot(x(IN,1), x(IN,2), 'ro', 'MarkerSize', 5);
hold on;

end
