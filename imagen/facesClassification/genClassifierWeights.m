% Faces classification 
clear all;
close all;

localPath = '/volatile/bernardng/data/imagen/';
netwPath = '/home/bn228083/';
addpath(genpath([netwPath,'matlabToolboxes/markSchmidtCode']));
addpath(genpath([netwPath,'matlabToolboxes/general']));
addpath(genpath([netwPath,'matlabToolboxes/libsvm/matlab']));

fid = fopen([localPath,'subjectLists/facesList.txt']);
nSubs = 58;
sublist = cell(nSubs,1);
for i = 1:nSubs
    sublist{i} = fgetl(fid);
end

nROIs = 491;
nFolds = 9;
options.Method = 'lbfgs'; 
options.optTol = 1e-6; 
options.progTol = 1e-6; 
options.order = -1;

method = 5;
subs = 1:10;

if method == 2
    load('scoreSVM');
elseif method == 3
    load('scoreRidge');
elseif method == 4
    load('scoreLASSO');
elseif method == 5
    load('scoreEN');
elseif method == 6
    load('scoreAnat');
elseif method == 7
    load('scoreAnatGroup');
end

wAcc = zeros(nROIs+1,nFolds,length(subs));
for sub = subs
    disp(['Subject ',int2str(sub)]);
    [features,labels] = loadFeatLabel(sublist{sub});
    [nSamp,nFeat] = size(features);
    [trainInd,testInd] = cvSeq(nSamp,nFolds);
    for fold = 1:nFolds
        X = features(trainInd{fold},:); Y = labels(trainInd{fold});
        if method == 1 % Logistic Regression
            funObj = @(w)LogisticLoss(w,X,Y);
            wInit = zeros(nFeat,1);
            wAcc(:,fold,sub) = minfunc(funObj,wInit,options);
            if median(X(1:8,:)*wAcc(:,fold,sub)) < 0
                wAcc(:,fold,sub) = -wAcc(:,fold,sub);
            end
        elseif method == 2 % Support Vector Machine
            cmd = ['-s 0 -t 0 -c ',num2str(cOptSVM(fold,sub)),' -q'];
            model = svmtrain(Y,X(:,1:end-1),cmd);
            wAcc(1:nROIs,fold,sub) = model.SVs'*model.sv_coef;
            wAcc(nFeat,fold,sub) = -model.rho;
            if median(X(1:8,:)*wAcc(:,fold,sub)) < 0
                wAcc(:,fold,sub) = -wAcc(:,fold,sub);
            end
        elseif method == 3 % l2-regularized Logistic Regression
            alpha = alphaOptRidge(fold,sub)*ones(nFeat,1);
            alpha(end) = 0;
            funObj = @(w)logisticLoss(w,X,Y);
            funObjL2 = @(w)penalizedl2(w,funObj,alpha);
            wInit = zeros(nFeat,1);
            wAcc(:,fold,sub) = minFunc(funObjL2,wInit,options);
            if median(X(1:8,:)*wAcc(:,fold,sub)) < 0
                wAcc(:,fold,sub) = -wAcc(:,fold,sub);
            end
        elseif method == 4 % Sparse Logistic Regression
            lambda = lambdaOptLASSO(fold,sub)*ones(nFeat,1);
            lambda(end) = 0;
            funObj = @(w)logisticLoss(w,X,Y);
            wInit = zeros(nFeat,1);
            wAcc(:,fold,sub) = L1GeneralProjection(funObj,wInit,lambda,options);
            if median(X(1:8,:)*wAcc(:,fold,sub)) < 0
                wAcc(:,fold,sub) = -wAcc(:,fold,sub);
            end
        elseif method == 5 % Logistic Regression with Elastic Net
            alphaOpt = alphaOptEN(fold,sub);
            lambda = lambdaOptEN(fold,sub)*ones(nFeat,1);
            lambda(end) = 0;
            funObj = @(w)logisticLoss(w,X,Y);
            funObjLap = @(w)penalizedl2(w,funObj,(1-alphaOpt)*lambdaOptEN(fold,sub));
            wInit = zeros(nFeat,1);
            wAcc(:,fold,sub) = L1GeneralProjection(funObjLap,wInit,alphaOpt*lambda,options);
            if median(X(1:8,:)*wAcc(:,fold,sub)) < 0
                wAcc(:,fold,sub) = -wAcc(:,fold,sub);
            end
        elseif method == 6 % Subject-specific Anatomical Laplacian Sparse Logistic Regression
            load([localPath,sublist{sub},'/dwi/K_anatDense_parcel500.mat']);
            L = Kanat;
            L = diag(sum(L))-L;
            alphaOpt = alphaOptAnat(fold,sub);
            lambda = lambdaOptAnat(fold,sub)*ones(nFeat,1);
            lambda(end) = 0;
            funObj = @(w)logisticLoss(w,X,Y);
            funObjLap = @(w)LaplacianBN(w,funObj,(1-alphaOpt)*lambdaOptAnat(fold,sub),L);
            wInit = zeros(nFeat,1);
            wAcc(:,fold,sub) = L1GeneralProjection(funObjLap,wInit,alphaOpt*lambda,options);
            if median(X(1:8,:)*wAcc(:,fold,sub)) < 0
                wAcc(:,fold,sub) = -wAcc(:,fold,sub);
            end
        elseif method == 7 % Group Anatomical Laplacian Sparse Logistic Regression
            load([localPath,sublist{sub},'/dwi/K_anatGroup_parcel500.mat']);
            L = Kanat;
            L = diag(sum(L))-L;
            alphaOpt = alphaOptAnatGroup(fold,sub);
            lambda = lambdaOptAnatGroup(fold,sub)*ones(nFeat,1);
            lambda(end) = 0;
            funObj = @(w)logisticLoss(w,X,Y);
            funObjLap = @(w)LaplacianBN(w,funObj,(1-alphaOpt)*lambdaOptAnatGroup(fold,sub),L);
            wInit = zeros(nFeat,1);
            wAcc(:,fold,sub) = L1GeneralProjection(funObjLap,wInit,alphaOpt*lambda,options);
            if median(X(1:8,:)*wAcc(:,fold,sub)) < 0
                wAcc(:,fold,sub) = -wAcc(:,fold,sub);
            end
        end
    end
end


