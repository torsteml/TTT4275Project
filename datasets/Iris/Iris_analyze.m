%% Load dataset
clear all;
alpha = 0.0009;
train_iterations = 1000;
plots = true;
iterateParams = true;
if(iterateParams)
    alpha = [0.00001:1/10000:0.01];
    train_iterations = [10, 100, 1000, 10000];
    plots = false;
end
for alp=1:length(alpha)
for iter=1:length(train_iterations)
if(iterateParams)
    disp("Alpha: " + alpha(alp));
    disp("Iterations: " + train_iterations(iter));
end
[training_set, testing_set, training_idx, testing_idx]  = load_iris();
N_train = size(training_set,1);
N_test = size(testing_set,1);
x=[training_set, ones(N_train,1)];
t=training_idx;
C=3; % Number of classes
D=4; % Feature size
Ds = ["Sepal Length","Sepal Width","Petal Length","Petal Width"];
%% Train
close all
% Preallocate
W = zeros(C,D+1); % W+wo
W_all = zeros(C,D+1, train_iterations(iter)); % 
grad_MSE_all = zeros(C,D+1, train_iterations(iter));
%wo = zeros(C,1); % Class offset
g = zeros(N_train, C);

% Calculate discriminant vector g
for train_iteration = 1:train_iterations(iter)
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
    W_all(:,:,train_iteration) = W;
    W = W - alpha(alp).*grad_MSE;
end
W_all(:,:,train_iteration) = W;
%% Test
x_test=[testing_set, ones(N_test,1)];
predicted = iris_predict(x_test,W);
% Create a ground truth vector of correct class labels
ground_truth = testing_idx(:,1)+testing_idx(:,2)*2+testing_idx(:,3)*3;
% Compare predicted labels with correct labels
correct = predicted == ground_truth; % 1=correct prediction, 0=wrong prediction

%% Calculate different measures
% Test set
[TP,TN,FP,FN]=calculate_testingNumbers(predicted, ground_truth);

[TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, ACC, F1] =...
    calculate_testingMeasures(TP, TN, FP, FN);

[TPR_tot, TNR_tot, PPV_tot, NPV_tot, FNR_tot, FPR_tot, FDR_tot, FOR_tot, ACC_tot, F1_tot] =...
    calculate_testingMeasures(sum(TP), sum(TN), sum(FP), sum(FN));
confusion_matrix = confusionmat(ground_truth, predicted)
% Training set
predicted_training = iris_predict(x,W);
ground_truth_training = training_idx(:,1)+training_idx(:,2)*2+training_idx(:,3)*3;
[TP_train,TN_train,FP_train,FN_train]=...
    calculate_testingNumbers(predicted_training, ground_truth_training);

[TPR_train, TNR_train, PPV_train, NPV_train, FNR_train, FPR_train, FDR_train, FOR_train, ACC_train, F1_train] =...
    calculate_testingMeasures(sum(TP_train), sum(TN_train), sum(FP_train), sum(FN_train));

confusion_matrix_train = confusionmat(ground_truth_training, predicted_training)

if(iterateParams)
    ACCs(iter,alp)=ACC_tot;
    disp(ACC_tot);
end


%% Show results W_all(2,1,:)
% x1 = [1:train_iterations];
if(plots)
%     subplot(2,2,d)
%     y1 = reshape(W_all(1,d,:),train_iterations,1);
%     y2 = reshape(W_all(2,d,:),train_iterations,1);
%     y3 = reshape(W_all(3,d,:),train_iterations,1);
%     plot(x1,y1,x1,y2,x1,y3);
% end
    title("Weights for " + Ds(d));
    xlabel("Iterations");
    ylabel("Weight");
%     y1 = reshape(grad_MSE_all(1,d,:),train_iterations,1);
%     y2 = reshape(grad_MSE_all(2,d,:),train_iterations,1);
%     y3 = reshape(grad_MSE_all(3,d,:),train_iterations,1);
%     plot(x1,y1,x1,y2,x1,y3);
% end
    title("Errors");
    xlabel("Iterations");
    ylabel("Error");
%% Plot confusion matrix
figure;
cm=confusionchart(confusion_matrix);
cm.Title = 'Confusion Matrix using Linear Classifier for Test Set';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
return
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
            y_dim = row;
            x_dim = col;
            % w1*x1+w2*x2+w0=0
            % x1 =- w2/w1*x2-w0/w1
            w1 = W(c,y_dim);
            w2 = W(c,x_dim);
            w0 = W(c,end);
            % y = a*x+b
            b = -w0/w1;
            alp = -(w1/w2);
            y = @(x) alp.*x+b;
            ys(c,:) = y(xs);
        end
        % Plot decision boundary lines
%         for c = 1:1
%         for c = 1:C
%             plot(xs,ys(c,:),'Color',color_map(c,:));
%         end
       % title(['Row: ' num2str(row) ', Col: ' num2str(col)]);
        hold off
    end
end
end
end
end

if(iterateParams)
    plot(alpha,ACCs)
    title("Plot of accuracy versus alpha");
    xlabel("Alpha");
    ylabel("Accuracy");
    legend(strtrim(cellstr(num2str(train_iterations'))'))
end

















