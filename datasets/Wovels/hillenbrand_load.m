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
    formant_matrix = [tbl.F0s, tbl.F1s, tbl.F2s, tbl.F3s];
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
% Create test matrix where each column is F0-F4
testing_set = zeros(test_len*numel(vowels),4);
ground_truth = zeros(test_len*numel(vowels),1);
for vowel_num = 1:numel(vowels)
    tbl = vowel_classes_test{vowel_num,1};
   testing_set(1+test_len*(vowel_num-1):(test_len*vowel_num),:) = ...
       [tbl.F0s, tbl.F1s, tbl.F2s, tbl.F3s]; 
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



%% 1c) Create single Gaussian class model for each class with only diagonal matrices 
gaussian_models_diag = cell(numel(vowels),1);
for row = 1:numel(vowels)
    mu=vowel_classes_train{row,2}; 
    sigma=diag(vowel_classes_train{row,3}).*eye(4);
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
%% Make scatter plots




%% Make histograms
x = formant_table.F0s(formant_table.talker=='m');
figure(1);
subplot(221);
hist(x,20);
% use 20 �bins�
set(gca,'XLim',[50 500]); % set x-axis limits between 50 & 500 Hz
title('adult males');