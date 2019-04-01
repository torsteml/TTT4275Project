%% Hillenbrand Load

formant_table = load_vowdata();

%% 1a) Find the sample mean and the sample covariance matrix for each class set.
train_len = 70;
test_len = 69;
% # Seperate into each vowel class
vowels = unique(formant_table.vowel);
vowel_classes = cell(numel(vowels),3);
vowel_classes_train = cell(numel(vowels),3); % data, mean, covariance
vowel_classes_test = cell(numel(vowels),3);

for row = 1:numel(vowels)
   vowel_classes{row,1} = formant_table(formant_table.vowel == vowels(row),:);
   vowel_classes_train{row,1} = vowel_classes{row,1}(1:70,:);
   vowel_classes_test{row,1} = vowel_classes{row,1}(71:139,:);
end

% ## Find mean and covariance for each vowel
for row = 1:numel(vowels)
    tbl = vowel_classes_train{row,1};
    formant_matrix = table2array(tbl(:,6:end));
    vowel_classes_train{row,2} = mean(formant_matrix); 
    vowel_classes_train{row,3} = cov(formant_matrix); 
end

%% 1b) Create single Gaussian class model for each class. 
gaussian_models = cell(numel(vowels),1);
for row = 1:numel(vowels)
    mu=vowel_classes_train{row,2}; 
    sigma=vowel_classes_train{row,3};
    gaussian_models{row} = gmdistribution(mu,sigma);
end

%% Classify training set
% Create test matrix where each column is F0-end
testing_set = zeros(test_len*numel(vowels),14);
ground_truth = zeros(test_len*numel(vowels),1);
for vowel_num = 1:numel(vowels)
    tbl = vowel_classes_test{vowel_num,1};
   testing_set(1+test_len*(vowel_num-1):(test_len*vowel_num),:) = ...
       table2array(tbl(:,6:end)); 
   ground_truth(1+test_len*(vowel_num-1):(test_len*vowel_num),:) = vowel_num;
end
% Classify
class_probabilities = zeros(size(testing_set,1),numel(vowels));

for vowel_num = 1:numel(vowels)
    class_probabilities(:,vowel_num) = pdf(gaussian_models{vowel_num,1},testing_set);
end

[~, predicted_classes] = max(class_probabilities,[],2);
% Compare predicted labels with correct labels
correct = predicted_classes == ground_truth; % 1=correct prediction, 0=wrong prediction
%% Evaluate errors. 
[TP,TN,FP,FN]=calculate_testingNumbers(predicted_classes, ground_truth);
[TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, ACC, F1] =...
    calculate_testingMeasures(TP, TN, FP, FN);
[TPR_tot, TNR_tot, PPV_tot, NPV_tot, FNR_tot, FPR_tot, FDR_tot, FOR_tot, ACC_tot, F1_tot] =...
    calculate_testingMeasures(sum(TP), sum(TN), sum(FP), sum(FN));
disp("Accuracy: " + ACC_tot);
disp("Error rate: " + (1-ACC_tot));

% #Confusion matrix
figure
cm = confusionchart(ground_truth,predicted_classes)
cm.Title = 'Confusion Matrix using single Gaussian with full Covariance Matrix';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%% 1c) Create single Gaussian class model for each class with only diagonal matrices 
gaussian_models_diag = cell(numel(vowels),1);
for row = 1:numel(vowels)
    mu=vowel_classes_train{row,2}; 
    sigma=diag(vowel_classes_train{row,3}).*eye(14);
    gaussian_models_diag{row} = gmdistribution(mu,sigma);
end

% Classify
class_probabilities_diag = zeros(size(testing_set,1),numel(vowels));

for vowel_num = 1:numel(vowels)
    class_probabilities_diag(:,vowel_num) = pdf(gaussian_models_diag{vowel_num,1},testing_set);
end

[~, predicted_classes_diag] = max(class_probabilities_diag,[],2);
% Compare predicted labels with correct labels
correct_diag = predicted_classes_diag == ground_truth; % 1=correct prediction, 0=wrong prediction

%% Evaluate errors. 
[TPd,TNd,FPd,FNd]=calculate_testingNumbers(predicted_classes_diag, ground_truth);
[TPRd, TNRd, PPVd, NPVd, FNRd, FPRd, FDRd, FORd, ACCd, F1d] =...
    calculate_testingMeasures(TPd, TNd, FPd, FNd);
[TPR_totd, TNR_totd, PPV_totd, NPV_totd, FNR_totd, FPR_totd, FDR_totd, FOR_totd, ACC_totd, F1_totd] =...
    calculate_testingMeasures(sum(TPd), sum(TNd), sum(FPd), sum(FNd));
disp("Accuracy diagonal: " + ACC_totd);
disp("Error rate diagonal: " + (1-ACC_totd));

% #Confusion matrix
figure
cm = confusionchart(ground_truth,predicted_classes_diag)
cm.Title = 'Confusion Matrix using single Gaussian with diagonal Covariance Matrix';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%% Oppgave 2

% Create gaussian mixture model for each class
mixtures = [2,3]; % Do one run for 2 mixtures for each class, and one for 3.
gaussian_mixture_models = cell(numel(vowels),numel(mixtures));
for row = 1:numel(vowels)
    tbl = vowel_classes_train{row,1};
    data_matrix = table2array(tbl(:,6:end));
    fprintf('Vowel %s: \n', char(vowels(row)));
    for m_num = 1:numel(mixtures)
        m=mixtures(m_num);
        fprintf('-m=%d\n',m);
        try
        gaussian_mixture_models{row,m_num} = fitgmdist(data_matrix,m,...
            'CovarianceType','diagonal','Options',statset('TolFun',1e-8),...
            'RegularizationValue',0.1,'Replicates',5);
        catch exception
            disp('There was an error fitting the Gaussian mixture model')
            error = exception.message
        end
    end
end

%% Classify
% Classify
class_probabilities_gmm = zeros(size(testing_set,1),numel(vowels),numel(mixtures));

for vowel_num = 1:numel(vowels)
    for m_num = 1:numel(mixtures)
        gmm = gaussian_mixture_models{vowel_num,m_num};
        if ~isempty(gmm)     
            class_probabilities_gmm(:,vowel_num,m_num) =...
                pdf(gmm,testing_set);
%             for x = 1:size(testing_set,1)
%             class_probabilities_gmm(x,vowel_num,m_num) =...
%                            sum(mvnpdf(testing_set(x,:), gmm.mu, gmm.Sigma));
%             end
        end
    end
end

[~, predicted_classes_gmm] = max(class_probabilities_gmm,[],2);
predicted_classes_gmm = [predicted_classes_gmm(:,:,1), predicted_classes_gmm(:,:,2)];
% Compare predicted labels with correct labels
correct_gmm=zeros(size(predicted_classes,1),numel(mixtures));
for m_num = 1:numel(mixtures)
    correct_gmm(:,m_num) = predicted_classes_gmm(:,m_num) == ground_truth; % 1=correct prediction, 0=wrong prediction
end

%% Evaluate errors

for m_num = 1:numel(mixtures)
    [TP,TN,FP,FN]=calculate_testingNumbers(predicted_classes_gmm(:,m_num), ground_truth);
    [TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, ACC, F1] =...
        calculate_testingMeasures(TP, TN, FP, FN);
    [TPR_tot, TNR_tot, PPV_tot, NPV_tot, FNR_tot, FPR_tot, FDR_tot, FOR_tot, ACC_tot, F1_tot] =...
        calculate_testingMeasures(sum(TP), sum(TN), sum(FP), sum(FN));
    disp("Accuracy gmm using " +mixtures(m_num) + " mixtures " + ACC_tot);
    disp("Error rate gmm: " + (1-ACC_tot));

    % #Confusion matrix
    figure
    cm = confusionchart(ground_truth,predicted_classes_gmm(:,m_num));
    cm.Title = ['Confusion Matrix using GMM with ' num2str(mixtures(m_num)) ' mixtures'];
    cm.RowSummary = 'row-normalized';
    cm.ColumnSummary = 'column-normalized'; 
    
end
















