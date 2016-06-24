function five_fold()
% 5-fold cross validation

    load usps_3_8.mat;
%----------------------------------------------------------------------------------------------------------------------------------
%------------------------------------------------------------bayes-----------------------------------------------------------------
%----------------------------------------------------------------------------------------------------------------------------------
    thresholds=[0.5,0.6,0.7,0.75,0.8,0.85,0.9];
    accuracy=0;
    for i=1:length(thresholds)
        [~,b]=nbayesclassifier(digits_data(201:1000,:),digits_label(201:1000,:),digits_data(1:200,:),digits_label(1:200,:),thresholds(1,i));
        if(b>accuracy)
            accuracy=b;
            s.threshold=thresholds(1,i);
        end
    end
    for i=1:length(thresholds)
        [~,b]=nbayesclassifier([digits_data(1:200,:);digits_data(401:1000,:)],[digits_label(1:200,:);digits_label(401:1000,:)],digits_data(201:400,:),digits_label(201:400,:),thresholds(1,i));
        if(b>accuracy)
            accuracy=b;
            s.threshold=thresholds(1,i);
        end
    end
    for i=1:length(thresholds)
        [~,b]=nbayesclassifier([digits_data(1:400,:);digits_data(601:1000,:)],[digits_label(1:400,:);digits_label(601:1000,:)],digits_data(401:600,:),digits_label(401:600,:),thresholds(1,i));
        if(b>accuracy)
            accuracy=b;
            s.threshold=thresholds(1,i);
        end
    end
     for i=1:length(thresholds)
        [~,b]=nbayesclassifier([digits_data(1:600,:);digits_data(801:1000,:)],[digits_label(1:600,:);digits_label(801:1000,:)],digits_data(601:800,:),digits_label(601:800,:),thresholds(1,i));
        if(b>accuracy)
            accuracy=b;
            s.threshold=thresholds(1,i);
        end
     end
    for i=1:length(thresholds)
        [~,b]=nbayesclassifier(digits_data(1:800,:),digits_label(1:800,:),digits_data(801:1000,:),digits_label(801:1000,:),thresholds(1,i));
        if(b>accuracy)
            accuracy=b;
            s.threshold=thresholds(1,i);
        end
    end
%----------------------------------------------------------------------------------------------------------------------------------
%------------------------------------------------------------linear----------------------------------------------------------------
%----------------------------------------------------------------------------------------------------------------------------------
    lambdas=[1e-4,0.01,0.1,0.5,1,5,10,100,1000,5000,100000];
	accuracy=0;
%    f=fopen('linear.xls','w');
    for i=1:length(lambdas)
        [~,b]=lsclassifier(digits_data(201:1000,:),digits_label(201:1000,:),digits_data(1:200,:),digits_label(1:200,:),lambdas(1,i));
        if(b>accuracy)
            accuracy=b;
            s.lambda=lambdas(1,i);
        end
    end
    for i=1:length(lambdas)
        [~,b]=lsclassifier([digits_data(1:200,:);digits_data(401:1000,:)],[digits_label(1:200,:);digits_label(401:1000,:)],digits_data(201:400,:),digits_label(201:400,:),lambdas(1,i));
        if(b>accuracy)
            accuracy=b;
            s.lambda=lambdas(1,i);
        end
    end
    for i=1:length(lambdas)
        [~,b]=lsclassifier([digits_data(1:400,:);digits_data(601:1000,:)],[digits_label(1:400,:);digits_label(601:1000,:)],digits_data(401:600,:),digits_label(401:600,:),lambdas(1,i));
        if(b>accuracy)
            accuracy=b;
            s.lambda=lambdas(1,i);
        end
    end
     for i=1:length(lambdas)
        [~,b]=lsclassifier([digits_data(1:600,:);digits_data(801:1000,:)],[digits_label(1:600,:);digits_label(801:1000,:)],digits_data(601:800,:),digits_label(601:800,:),lambdas(1,i));
        if(b>accuracy)
            accuracy=b;
            s.lambda=lambdas(1,i);
        end
     end
    for i=1:length(lambdas)
        [~,b]=lsclassifier(digits_data(1:800,:),digits_label(1:800,:),digits_data(801:1000,:),digits_label(801:1000,:),lambdas(1,i));
        if(b>accuracy)
            accuracy=b;
            s.lambda=lambdas(1,i);
        end
    end
%----------------------------------------------------------------------------------------------------------------------------------
%------------------------------------------------------------svm-------------------------------------------------------------------
%----------------------------------------------------------------------------------------------------------------------------------
    Cs=[1,10,100,1000];
	sigmas=[0.01,0.1,1,10,100];
	accuracy=0;
    f=fopen('svm.xls','w');
    for i=1:length(Cs)
        data=digits_data(201:1000,:);
        d=sum(pdist(data, 'euclidean').*pdist(data, 'euclidean'))*2/800/800;
        for j=1:length(sigmas)
            [~,b]=softsvm(data,digits_label(201:1000,:),digits_data(1:200,:),digits_label(1:200,:),sigmas(1,j)*d,Cs(1,i));
            fprintf(f,'%f\t',b);
            if(b>accuracy)
                accuracy=b;
                s.sigma=sigmas(1,j);
                s.C=Cs(1,i);
            end
        end
        fprintf(f,'\n');
    end
    fprintf(f,'\n');
    for i=1:length(Cs)
        data=[digits_data(1:200,:);digits_data(401:1000,:)];
        d=sum(pdist(data, 'euclidean').*pdist(data, 'euclidean'))*2/800/800;
        for j=1:length(sigmas)
            [~,b]=softsvm(data,[digits_label(1:200,:);digits_label(401:1000,:)],digits_data(201:400,:),digits_label(201:400,:),sigmas(1,j)*d,Cs(1,i));
            fprintf(f,'%f\t',b);
            if(b>accuracy)
                accuracy=b;
                s.sigma=sigmas(1,j);
                s.C=Cs(1,i);
            end
        end
        fprintf(f,'\n');
    end
    fprintf(f,'\n');
    for i=1:length(Cs)
        data=[digits_data(1:400,:);digits_data(601:1000,:)];
        d=sum(pdist(data, 'euclidean').*pdist(data, 'euclidean'))*2/800/800;
        for j=1:length(sigmas)
            [~,b]=softsvm(data,[digits_label(1:400,:);digits_label(601:1000,:)],digits_data(401:600,:),digits_label(401:600,:),sigmas(1,j)*d,Cs(1,i));
            fprintf(f,'%f\t',b);
            if(b>accuracy)
                accuracy=b;
                s.sigma=sigmas(1,j);
                s.C=Cs(1,i);
            end
        end
        fprintf(f,'\n');
    end
    fprintf(f,'\n');
    for i=1:length(Cs)
        data=[digits_data(1:600,:);digits_data(801:1000,:)];
        d=sum(pdist(data, 'euclidean').*pdist(data, 'euclidean'))*2/800/800;
        for j=1:length(sigmas)
            [~,b]=softsvm(data,[digits_label(1:600,:);digits_label(801:1000,:)],digits_data(601:800,:),digits_label(601:800,:),sigmas(1,j)*d,Cs(1,i));
            fprintf(f,'%f\t',b);
            if(b>accuracy)
                accuracy=b;
                s.sigma=sigmas(1,j);
                s.C=Cs(1,i);
            end
        end
        fprintf(f,'\n');
    end
    fprintf(f,'\n');
    for i=1:length(Cs)
        data=digits_data(1:800,:);
        d=sum(pdist(data, 'euclidean').*pdist(data, 'euclidean'))*2/800/800;
        for j=1:length(sigmas)
            [~,b]=softsvm(data,digits_label(1:800,:),digits_data(801:1000,:),digits_label(801:1000,:),sigmas(1,j)*d,Cs(1,i));
            fprintf(f,'%f\t',b);
            if(b>accuracy)
                accuracy=b;
                s.sigma=sigmas(1,j);
                s.C=Cs(1,i);
            end
        end
        fprintf(f,'\n');
    end
    fprintf(f,'\n');
    fclose(f);
    save('_parameters.mat','-struct','s');