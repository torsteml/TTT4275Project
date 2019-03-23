%% Hillenbrand Load

formant_table = load_vowdata();
%% Make scatter plots




%% Make histograms
x = formant_table.F0s(formant_table.talker=='m');
figure(1);
subplot(221);
hist(x,20);
% use 20 “bins”
set(gca,'XLim',[50 500]); % set x-axis limits between 50 & 500 Hz
title('adult males');