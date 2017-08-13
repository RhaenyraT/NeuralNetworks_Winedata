%IMPORT WINE DATA
datawhite = importdata('winequality-white.csv');
Index = (datawhite.data(:,12)==4 | datawhite.data(:,12)==5 | datawhite.data(:,12)==6);
X=datawhite.data(Index,1:(end-1))';
T=datawhite.data(Index,end);
T(T~=4) = 0; T(T==4) = 1;
Target = T';
[Train_index,~,Test_index] = dividerand(size(X,2),0.8,0,0.20);
X_Training = X(:,Train_index); Target_Training = Target(:,Train_index);
X_Test = X(:,Test_index); Target_Test = Target(:,Test_index);
%%
%FEED FORWARD WITHOUT PCA
hidden_neurons=10;
test_performance=zeros(size(hidden_neurons,2),1);
test_errors=zeros(size(hidden_neurons,2),764);
ccr=zeros(size(hidden_neurons,2),1);

for r=1:size(hidden_neurons,2)
     [tr, test_Outputs, test_errors, test_performance ] = neuralnetsclass(X_Training,Target_Training,hidden_neurons(r),X_Test,Target_Test);
     plotconfusion(Target_Test,test_Outputs)
     ccr = sum(Target_Test==test_Outputs)/length(test_Outputs);
end
%%
% FIND PRINCIPAL COMPONENTS 
X_Training= bsxfun(@minus, threes, mean(X_Training));
C = cov(X_Training');
[eig_Vec,eig_Val] = eig(C);
eig_Val = diag(eig_Val);
[eig_Val,ind]=sort(eig_Val,'descend');
eig_Vec = eig_Vec(:,ind);
    
    figure, hold on;
    plot(eig_Val);
    xlabel('Index');
    ylabel('Eigenvalue');
    title('PCA Eigenvalue Plot');
      
    cumulated_EigVal = cumsum(eig_Val/sum(eig_Val));
    figure, hold on;
    plot(cumulated_EigVal);
    xlabel('Number of Principal Components');
    ylabel('Fraction of Variance Captured');
    title('PCA Variance Plot');
    
% RECONSTRUCTING TRAINING AND TEST DATA IN NEW DIMENSION    
 eig_VecT= (eig_Vec(:,1:8))';
Newdata=eig_VecT*X_Training;
NewTestData=eig_VecT*X_Test;


% FEED FORWARD WITH PCA
hidden_neurons=100;
test_performance=zeros(size(hidden_neurons,2),1);
test_errors=zeros(size(hidden_neurons,2),764);
ccr=zeros(size(hidden_neurons,2),1);
for r=1:size(hidden_neurons,2)
     [tr, test_Outputs, test_errors, test_performance ] = neuralnetsclass(Newdata,Target_Training,hidden_neurons(r),NewTestData,Target_Test);
     plotconfusion(Target_Test,test_Outputs)
     ccr = sum(Target_Test==test_Outputs)/length(test_Outputs);
end
