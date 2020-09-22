function [X_train y_train X_test y_test] = holdout(X,y,m,percent)
perm = randperm(m);
splitpoint = floor(m*percent);

X_train = X(perm(1:splitpoint),:);
y_train = y(perm(1:splitpoint));
X_test = X(perm(splitpoint+1:end),:);
y_test = y(perm(splitpoint+1:end));
end
