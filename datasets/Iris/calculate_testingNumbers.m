function [TP,TN,FP,FN]=calculate_testingNumbers(predicted, ground_truth)
C = 3;
% True positives
TP = zeros(C,1);
for c = 1:C
   TP(c) = nnz(predicted(ground_truth==c)==c);
end
% True negatives
TN = zeros(C,1);
for c = 1:C
   TN(c) = nnz(predicted(ground_truth~=c)~=c);
end
% False positives
FP = zeros(C,1);
for c = 1:C
   FP(c) = nnz(predicted(ground_truth~=c)==c);
end
% False negatives
FN = zeros(C,1);
for c = 1:C
   FN(c) = nnz(predicted(ground_truth==c)~=c);
end

end