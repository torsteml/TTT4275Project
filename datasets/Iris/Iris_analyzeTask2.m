clear all;
close all;

[training_set, testing_set, training_idx, testing_idx]  = load_iris();

%% Plot histograms

%for p=1:length(training_set(1,:))
 %   subplot(2,2,p);
  %  histogram(training_set(:,p),50);
%end

complete_set = [training_set; testing_set];
complete_idx = [training_idx; testing_idx];

nBins = 40;
hold on;
%sepal length
    %Setosa
    histPlot = complete_set(:,1).*complete_idx(:,1);
    histPlotReal = histPlot(histPlot~=0);
    histogram(histPlotReal,nBins);
    %versicolour
    histPlot = complete_set(:,1).*complete_idx(:,2);
    histPlotReal = histPlot(histPlot~=0);
    histogram(histPlotReal,30);
    %virginica
    histPlot = complete_set(:,1).*complete_idx(:,3);
    histPlotReal = histPlot(histPlot~=0);
    histogram(histPlotReal,30);
%sepal width
%petal length
%petal width

