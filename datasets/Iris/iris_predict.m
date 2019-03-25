function class = iris_predict(x,W)
C=3;
x_rows = size(x,1);    
g = zeros(x_rows, C);
    for i = 1:C
        for j = 1:x_rows
           g(j,i)= W(i,:)*x(j,:)';
        end
    end
    g = sigmoid(g);
    [~,class] =max(g,[],2);
end