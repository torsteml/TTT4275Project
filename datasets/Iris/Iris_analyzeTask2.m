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

features = ["Sepal Length", "Sepal Width", "Petal length", "Petal Width"];
classes = ["Setosa", "Versicolour", "Virginica"];
for feature=1:length(complete_set(1,:))
    subplot(2,2,feature);
    hold on;
    for class=1:length(complete_idx(1,:))
        histPlot = complete_set(:,feature).*complete_idx(:,class);
        histPlotReal = histPlot(histPlot~=0);
        histogram(histPlotReal,nBins);
    end
    legend(classes);
    hold off;
    title(features(feature));
end

