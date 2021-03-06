function [ypred,accuracy] = lsclassifier(traindata, trainlabel,testdata, testlabel, lambda)
% param traindata : the train data set
% param trainlabel : the train data ground truth label, 0 means digit_3
% param testdata : test data set
% param testlabel : test data label
% param lambda : L2 Regularization parameter
% ret ypred : the test result
% ret accuracy : test set accuracy=(pridict(i)==testlabel(i))/testsize
%{
    min( Xw-y)^2 + lambda*w'*w
    = (Xw-y)'*(Xw-y) + lambda*w'*w
    = w'*(X'X+lambda*I)*w -2y'Xw +y'y
%}
    [dim,~]=size(traindata);
    X=[ones(dim,1),traindata];
    [~,I]=size(X);
    % optimize the function
    w=quadprog(2*(X'*X+lambda*eye(I)),-2*trainlabel'*X);
    % directly calculate the soluation of the function
    %w_l=(X'*X+lambda*eye(I))\(X'*trainlabel);
    %w_l=pinv(X)*trainlabel;
    %disp([w,w_l]);
    
    [n,~]=size(testdata);
    test_out=[ones(n,1),testdata]*w;
    %disp(test_out);
    ypred=test_out(:,1)>0.5;
    test_label=~(testlabel(:,1)==0);
    accuracy=(sum(ypred==test_label)/n);
%     disp(accuracy);
%     
%     train_out=[ones(dim,1),traindata]*w;
%     train_classifily=train_out(:,1)<0.5;
%     train_label=trainlabel(:,1)==0;
%     train_accuracy=(sum(train_label==train_classifily)/dim);
%     disp(train_accuracy);