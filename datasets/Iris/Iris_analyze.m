%% Load dataset
[training_set, testing_set] = load_iris();
N_train = size(training_set,1);
N_test = size(testing_set,1);
x=training_set;
C=3;
D=1;
alpha = 0.5;
%% Train

% Preallocate
W = zeros(C,D); % 
wo = zeros(C,D); % Class offset

% Create discriminant vector g
%x_vec = [training_set', ones(N_train,1)];
%g = dot([W, wo],(x_vec'));
g = W*x+wo;
gi = sigmoid(g);
% Create gradient MSE
grad_MSE = zeros(N_train,D);
MSE_1 = (g-t).*(1-g);
MSE_2 = dot(MSE_1,x');
grad_MSE = sum(MSE_2);

% Update W
W = circshift(W,1) - alpfa.*grad_MSE;

%% Test


%% Show results