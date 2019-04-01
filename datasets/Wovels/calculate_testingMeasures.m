function [TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, ACC, F1] =...
    calculate_testingMeasures(TP, TN, FP, FN)
    % True positive rate, sensitivity
    TPR = TP./(TP+FN);
    % True negative rate, specificity
    TNR = TN./(TN+FP);
    % Positive preditictive rate, precision
    PPV = TP./(TP+FP);
    % Negative preditictive rate
    NPV = TN./(TN+FN);
    % False negative rate, miss rate
    FNR = 1-TPR;
    % False positive rate, fall-out
    FPR = 1-TNR;
    % False discovery rate
    FDR = 1-PPV;
    % False ommission rate
    FOR = 1-NPV;
    % Accuracy
    ACC = (TP+TN)./(TP+TN+FP+FN);
    % F1 score
    %  - The harmonic mean of precision and sensitivity
    F1 = 2.*(PPV.*TPR)./(PPV+TPR);
end