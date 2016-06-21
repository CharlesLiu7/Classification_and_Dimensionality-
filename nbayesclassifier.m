function [ypred,accuracy] = nbayesclassifier(traindata,trainlabel, testdata, testlabel, threshold)
% param traindata : the train data set
% param trainlabel : the train data ground truth label, 0 means digit_3
% param testdata : test data set
% param testlabel : test data label
% param threshold : the belief of probability to be true
% ret ypred : the test result
% ret accuracy : test set accuracy=(pridict(i)==testlabel(i))/testsize
    
    COLOR=256;
    SHIFT=0.000001;
    
    logical_label_train=trainlabel(:,1)>=0 & trainlabel(:,1)<=0;
    % m - train data number of 3
    m=sum(logical_label_train);
    [n,~]=size(traindata);
    % set traindat belongs to {0, 1}
    bool_traindata=(traindata(:,:)>(COLOR/2-1));
    prior_3=logical_label_train'*double(bool_traindata)/m;
    prior_8=(~logical_label_train')*double(bool_traindata)/(n-m);
    %disp(cat(1,prior_3,prior_8));
    
    bool_testdata=testdata(:,:)>(COLOR/2-1);
    % P(Ck | X1, X2...Xn) calculate the log value
    prior_d3=bool_testdata*log(prior_3'+SHIFT)+(~bool_testdata)*log(1-prior_3'+SHIFT)+log(m/n);
    prior_d8=bool_testdata*log(prior_8'+SHIFT)+(~bool_testdata)*log(1-prior_8'+SHIFT)+log(1-m/n);
    [dim,~]=size(testdata);
    logical_label_test=testlabel(:,1)>=0 & testlabel(:,1)<=0;
    ypred=prior_d3*threshold>prior_d8;
    %disp(cat(2,prior_d3,prior_d8,logical_label_test));
    %disp(sum((prior_d3*threshold>prior_d8)==logical_label_test));
    accuracy=(sum((prior_d3*threshold>prior_d8)==logical_label_test))/dim;