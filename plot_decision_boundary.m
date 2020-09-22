function plot_decision_boundary(Theta1,Theta2, X, y)
    % Set min and max values and give it some padding
    %x_min = min(X(1, :)) - 6;
    %x_max = max(X(1, :)) + 1;
    %y_min = min(X(2, :)) - 3;
    %y_max = max(X(2, :)) + 4;
    x_min = min(X(1, :)) - 7;
    x_max = max(X(1, :)) + 8;%+5
    y_min = min(X(2, :)) - 8;
    y_max = max(X(2, :)) + 7;
    h = 0.01;
    % Generate a grid of points with distance h between them
    xx = [x_min:h:x_max];
    yy = [y_min:h:y_max];
    % Predict the function value for the whole grid
    [X1,Y1] = meshgrid(xx,yy);
    v=[X1(:) Y1(:)];
    Z = predict(Theta1, Theta2, v);
    Z = reshape(Z,size(X1));
    % Plot the contour and examples
   
    figure(); % open a new figure window
     
    contour(X1,Y1,Z);%,'LineColor','b','LineWidth',3);
    hold on;
    %map = [0 0 0.6;0.4 0 0.4];
    %map = [0.7 0 0.9;0.1 0 0.9];
    %colormap(map);

    ylabel('x2');
    xlabel('x1');
    hold on;
    IP = (y == 1);%Indices positivos
    IN = (y == 0);%Indices negativos
    plot(X(IP,1), X(IP,2), 'bo', 'MarkerSize', 5);
    hold on;
    plot(X(IN,1), X(IN,2), 'ro', 'MarkerSize', 5);
    hold on;
end
