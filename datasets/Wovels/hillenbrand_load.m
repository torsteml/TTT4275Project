%% Hillenbrand Load

formant_table = load_vowdata();

%% 1a) Find the sample mean and the sample covariance matrix for each class set.

% # Seperate into each vowel class
vowels = unique(formant_table.vowel);
vowel_classes = cell(numel(vowels),3); % data, mean, covariance
for row = 1:numel(vowels)
   vowel_classes{row,1} = formant_table(formant_table.vowel == vowels(row),:); 
end

% ## Find mean for each vowel
for row = 1:numel(vowels)
    tbl = vowel_classes{row,1};
    formant_matrix = [tbl.F0s, tbl.F1s, tbl.F2s, tbl.F3s];
    vowel_classes{row,2} = mean(formant_matrix); 
    vowel_classes{row,3} = cov(formant_matrix); 
end





%% Make scatter plots




%% Make histograms
x = formant_table.F0s(formant_table.talker=='m');
figure(1);
subplot(221);
hist(x,20);
% use 20 �bins�
set(gca,'XLim',[50 500]); % set x-axis limits between 50 & 500 Hz
title('adult males');