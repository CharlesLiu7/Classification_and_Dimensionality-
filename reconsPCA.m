function [proj_matrix,recons_data,recons_error]=reconsPCA(train_data,test_data, ground_truth,threshold)
% param train_data : the train data set
% param test_data : test data set
% param ground_truth : the test data ground truth
% param threshold : select first m vector s.t.m/all>=threshold>m-1/all
% ret proj_matrix : the project matrix
% ret recons_data : PCA the test_data result
% recons_error : test_data error

    %S=cov(train_data);
    [n,~]=size(train_data);
    S=train_data'*train_data/n;
    [eigenvectors,eigenvalues]=eig(S);
    m_sum=0;
    eigen_sum=trace(eigenvalues);
    for i=2500:-1:1
        m_sum=m_sum+eigenvalues(i,i);
        if((m_sum/eigen_sum)>=threshold)
            break;
        end
    end
    P=fliplr(eigenvectors(:,i:2500));
    
    proj_matrix=(P')*(test_data');
    recons_data=(P*proj_matrix)';
    error_diff=recons_data-ground_truth;
    recons_error=sum(error_diff.^2,2)/2500;
    save('recons_error.mat','recons_error');
    
    
