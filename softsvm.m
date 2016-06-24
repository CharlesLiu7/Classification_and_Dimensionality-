function [ypred,accuracy] = softsvm(traindata, trainlabel,testdata, testlabel, sigma, C)
% param traindata : the train data set
% param trainlabel : the train data ground truth label, 0 means digit_3
% param testdata : test data set
% param testlabel : test data label
% param sigma : the RBF kernel function parameter
% param C : soft margin SVM control parameter
% ret ypred : the test result
% ret accuracy : test set accuracy=(pridict(i)==testlabel(i))/testsize

    threshold=1e-10;
    y_train=trainlabel*2-1;
    [n,~]=size(traindata);
    if(sigma==0)
        H=(y_train*y_train').*(traindata*traindata');
    else
        H=(y_train*y_train').*exp(-squareform( pdist(traindata, 'euclidean').*pdist(traindata, 'euclidean'))/(sigma^2));
    end
    
    alpha=quadprog(H,-ones(n,1),[],[],y_train',0,0*ones(n,1),C*ones(n,1));
    indexs=(find(alpha>threshold));
    index=indexs(1,1);
    b=y_train(index,1)-sum(alpha.*y_train.*exp(-pdist2(traindata,traindata(index,:),'euclidean').*pdist2(traindata,traindata(index,:),'euclidean')/(sigma^2)));
    
    [m_test,~]=size(testdata);
    y_test=sum(repmat(alpha.*y_train,1,m_test).*exp(-pdist2(traindata,testdata,'euclidean').*pdist2(traindata,testdata,'euclidean')/(sigma^2)))+b;
    ypred=(y_test>0)';
    y_testlabel=~(testlabel==0);
    accuracy=sum(ypred==y_testlabel)/m_test;
    
    
