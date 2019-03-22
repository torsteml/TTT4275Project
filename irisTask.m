  %7. Attribute Information:
   %1. sepal length in cm
   %2. sepal width in cm
   %3. petal length in cm
   %4. petal width in cm

class1 = load("datasets/Iris/class_1",'-ascii');
class2 = load("datasets/Iris/class_2",'-ascii');
class3 = load("datasets/Iris/class_3",'-ascii');

%****Task 1****
%****a)****

Ntrain=30;
Ntest=20;

%Splits into training and testing
%0-30:setosa 31-60:versicolour 61-90:virginica
classesTrain = [class1(1:Ntrain,:);class2(1:Ntrain,:);class3(1:Ntrain,:)];
%0-20:setosa 21-40:versicolour 41-60:virginica
classesTest = [class1(1:Ntest,:);class2(1:Ntest,:);class3(1:Ntest,:)];

%Keeps track of flowers and orders by zeros and ones matrix
namesToClassesTrain = zeros(Ntrain*3,3);
    namesToClassesTrain(1:Ntrain,1) = 1; %setosa
    namesToClassesTrain(Ntrain+1:2*Ntrain,2) = 1; %versicolour
    namesToClassesTrain(2*Ntrain+1:3*Ntrain,3) = 1; %virginica
namesToClassesTest = zeros(Ntest*3,3);
    namesToClassesTest(1:Ntest,1) = 1; %setosa
    namesToClassesTest(Ntest+1:2*Ntest,2) = 1; %versicolour
    namesToClassesTest(2*Ntest+1:3*Ntest,3) = 1; %virginica
 
x = classesTrain(1,:);
%Preallocate
C = 3;
D = 1;
W = zeros(C,D);
W0 = zeros(C,D); 

g = W*x+W0;
gi = sigmoid(g);

MSE = zeros(Ntrain,D);

