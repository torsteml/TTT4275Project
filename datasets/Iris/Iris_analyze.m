%% Load dataset
[training_set, testing_set, training_idx, testing_idx]  = load_iris();
N_train = size(training_set,1);
N_test = size(testing_set,1);
x=[training_set, ones(N_train,1)];
t=training_idx;
C=3; % Number of classes
D=4; % Feature size
alpha = 0.0009;
train_iterations = 1000;
%% Train
close all
% Preallocate
W = zeros(C,D+1); % W+wo
W_all = zeros(C,D+1, train_iterations); % 
grad_MSE_all = zeros(C,D+1, train_iterations);
%wo = zeros(C,1); % Class offset
g = zeros(N_train, C);

% Calculate discriminant vector g
for train_iteration = 1:train_iterations
    for i = 1:C
        for j = 1:N_train
           g(j,i)= W(i,:)*x(j,:)';
        end
    end
    %%
    g_old = g;
    g = sigmoid(g);
    %%
    % Create gradient MSE
    MSE_1 = (g-t).*g.*(1-g);
    grad_MSE = MSE_1'*x;
    grad_MSE_all(:,:,train_iteration) = grad_MSE;
    % for k = 1:N_train
    %     MSE_2 = MSE_1(k,:)'*x(k,:);
    %     grad_MSE = grad_MSE+MSE_2;
    % end
    %%
    % Update W
    %W = circshift(W,1) - alpfa.*grad_MSE;
    W_all(:,:,train_iteration) = W;
    W = W - alpha.*grad_MSE;
end
W_all(:,:,train_iteration) = W;
%% Test


%% Show results W_all(2,1,:)
x1 = [1:train_iterations];
for d = 1:D
    subplot(2,2,d)
    y1 = reshape(W_all(1,d,:),train_iterations,1);
    y2 = reshape(W_all(2,d,:),train_iterations,1);
    y3 = reshape(W_all(3,d,:),train_iterations,1);
    plot(x1,y1,x1,y2,x1,y3);
end
figure;
for d = 1:D
    subplot(2,2,d)
    y1 = reshape(grad_MSE_all(1,d,:),train_iterations,1);
    y2 = reshape(grad_MSE_all(2,d,:),train_iterations,1);
    y3 = reshape(grad_MSE_all(3,d,:),train_iterations,1);
    plot(x1,y1,x1,y2,x1,y3);
end

%% Visualize model
fwidth = 0.5;
fheight = 0.5;
figure('Units', 'normalized','Position',[(1-fwidth)/2,(1-fheight)/2,fwidth,fheight]); % Center figure
color_map = brewermap(3,'Set1');
marker_size = 4;
dimensions = {'Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'};
data=training_set;
% Separate set into classes
class1 = data(training_idx(:,1)==1,:);
class2 = data(training_idx(:,2)==1,:);
class3 = data(training_idx(:,3)==1,:);
for row = 1:D
    for col = 1:D
        % Write feature name in the diagonal 
        if col == row
            ax = subplot(D,D,(row-1)*D+col);
            text(0.1,0.5,dimensions(row),'Units','normalized','Interpreter','latex');
            ax.XAxis.Visible='off';
            ax.YAxis.Visible='off';
            continue;
        end
        
        % Create scatter plot for each class 
        subplot(D,D,(row-1)*D+col);
        scatter(class1(:,col),class1(:,row),marker_size,color_map(1,:),'filled');
        hold on
        scatter(class2(:,col),class2(:,row),marker_size,color_map(2,:),'filled');
        scatter(class3(:,col),class3(:,row),marker_size,color_map(3,:),'filled');
        % Create decision boundary line
        decisionLine_points = 10;
        ax = gca;
        xLim = ax.XLim;
        xs = linspace(xLim(1),xLim(2),decisionLine_points);
        ys = zeros(C,decisionLine_points);
        for c = 1:C
            x_dim = col;
            y_dim = row;
            % w1*x+w2*y+w0=0
            w1 = W(c,x_dim);
            w2 = W(c,y_dim);
            w0 = W(c,end);
            % y = a*x+b
            b = -w0/w2;
            a = -(w1/w2);
            y = @(x) a.*x+b;
            ys(c,:) = y(xs);
        end
        % Plot decision boundary lines
%         for c = 1:C
%             plot(xs,ys(c,:),'Color',color_map(c,:));
%         end
        title(['Row: ' num2str(row) ', Col: ' num2str(col)]);
        hold off
    end
end

















