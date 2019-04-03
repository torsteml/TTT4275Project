function [training_set, testing_set, training_idx, testing_idx] = load_iris(class_path, Ntrain, Ntest)

if nargin < 1
   class_path = 'datasets/Iris/'; 
end
if nargin < 2
    Ntrain = 30;
end
if nargin < 3
    Ntest = 20;
end

%7. Attribute Information:
   %1. sepal length in cm
   %2. sepal width in cm
   %3. petal length in cm
   %4. petal width in cm

class1 = load([class_path 'class_1.dat'],'-ascii');
class2 = load([class_path 'class_2.dat'],'-ascii');
class3 = load([class_path 'class_3.dat'],'-ascii');

%Splits into training and testing
%0-30:setosa 31-60:versicolour 61-90:virginica
classesTrain = [class1(1:Ntrain,:);class2(1:Ntrain,:);class3(1:Ntrain,:)];
%%classesTrain = [class1(end-Ntrain+1:end,:);class2(end-Ntrain+1:end,:);class3(end-Ntrain+1:end,:)];

%0-20:setosa 21-40:versicolour 41-60:virginica
classesTest = [class1(end-Ntest+1:end,:);class2(end-Ntest+1:end,:);class3(end-Ntest+1:end,:)];
%%classesTest = [class1(1:Ntest,:);class2(1:Ntest,:);class3(1:Ntest,:)];

%Keeps track of flowers and orders by zeros and ones matrix
namesToClassesTrain = zeros(Ntrain*3,3);
    namesToClassesTrain(1:Ntrain,1) = 1; %setosa
    namesToClassesTrain(Ntrain+1:2*Ntrain,2) = 1; %versicolour
    namesToClassesTrain(2*Ntrain+1:3*Ntrain,3) = 1; %virginica
namesToClassesTest = zeros(Ntest*3,3);
    namesToClassesTest(1:Ntest,1) = 1; %setosa
    namesToClassesTest(Ntest+1:2*Ntest,2) = 1; %versicolour
    namesToClassesTest(2*Ntest+1:3*Ntest,3) = 1; %virginica
    
%classesTrain(:,3) = [];
classesTrain(:,2) = [];
%classesTrain(:,1) = [];
training_set = classesTrain;
%classesTest(:,3) = [];
classesTest(:,2) = [];
%classesTest(:,1) = [];
testing_set = classesTest;

training_idx = namesToClassesTrain;
testing_idx = namesToClassesTest;

end