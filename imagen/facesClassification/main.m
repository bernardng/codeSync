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

nFolds = 9;
niFolds = 8;
options.Method = 'lbfgs'; 
options.optTol = 1e-6; 
options.progTol = 1e-6; 
options.order = -1;

method = 1;
subs = 1:10;

score = zeros(nFolds,length(subs));
cOpt = zeros(nFolds,length(subs));
lambdaOpt = zeros(nFolds,length(subs));
alphaOpt = zeros(nFolds,length(subs));
% matlabpool(6);
for sub = subs
    disp(['Subject ',int2str(sub)]);
    [features,labels] = loadFeatLabel(sublist{sub});
%     Krest = load([localPath,sublist{sub},'/restfMRI/K_rest_anatDense_parcel500_quic335_cv.mat']);
%     L = Krest.Krest;
%     L = abs(L);
%     L = diag(sum(L))-L;
%     Krest = load([localPath,sublist{sub},'/restfMRI/K_rest_parcel500_quic335_cv.mat']);
%     L = Krest.Krest;
%     L = abs(L);
%     L = diag(sum(L))-L;
%     Kanat = load([localPath,sublist{sub},'/dwi/K_anatDense_parcel500.mat']);
%     L = Kanat.Kanat;
%     L = diag(sum(L))-L;
    
    [nSamp,nFeat] = size(features);
    % Cross validation
    [trainInd,testInd] = cvSeq(nSamp,nFolds);
    for fold = 1:nFolds
        X = features(trainInd{fold},:); Y = labels(trainInd{fold});
        [itrainInd,itestInd] = cvSeq(size(X,1),niFolds);
        if method == 1 % Logistic Regression
            funObj = @(w)LogisticLoss(w,X,Y);
            wInit = zeros(nFeat,1);
            w = minfunc(funObj,wInit,options);
            label = sign(features(testInd{fold},:)*w*median(X(1:8,:)*w));
            score(fold,sub) = mean(label==labels(testInd{fold}));
        elseif method == 2 % Support Vector Machine
            nGridPts = 6;
            cRange = logspace(-4,1,nGridPts);
            % Internal cross validation for parameter selection            
            iscore = zeros(niFolds,nGridPts);
            for ifold = 1:niFolds
                Xi = X(itrainInd{ifold},:); Yi = Y(itrainInd{ifold});
                for c = 1:nGridPts
                    cmd = ['-s 0 -t 0 -c ',num2str(cRange(c)),' -q'];
                    model = svmtrain(Yi,Xi(:,1:end-1),cmd);   
                    w = model.SVs'*model.sv_coef;
                    w = [w;-model.rho];
                    label = sign(X(itestInd{ifold},:)*w*median(X(1:8,:)*w));
%                     [label,accuracy] = svmpredict(Y(itestInd{ifold}),X(itestInd{ifold},1:end-1),model);
                    iscore(ifold,c) = mean(label==Y(itestInd{ifold}));
                end
            end
            iscoreAve = mean(iscore);
            [iscoreMax,ind] = max(iscoreAve);
            cOpt(fold,sub) = max(cRange(iscoreAve==iscoreMax));
%             cOpt = cRange(ind);
            cmd = ['-s 0 -t 0 -c ',num2str(cOpt(fold,sub)),' -q'];
            model = svmtrain(Y,X(:,1:end-1),cmd);
            w = model.SVs'*model.sv_coef; 
            w = [w;-model.rho];
            label = sign(features(testInd{fold},:)*w*median(X(1:8,:)*w));
%             [label,accuracy] = svmpredict(labels(testInd{fold}),features(testInd{fold},1:end-1),model);
            score(fold,sub) = mean(label==labels(testInd{fold}));            
        elseif method == 3 % l2-regularized Logistic Regression
            nGridPts = 6;
            alphaRange = logspace(-4,1,nGridPts);
            % Internal cross validation for parameter selection            
            iscore = zeros(niFolds,nGridPts);
            for ifold = 1:niFolds
                Xi = X(itrainInd{ifold},:); Yi = Y(itrainInd{ifold});
                for l = 1:nGridPts
                    alpha = alphaRange(l)*ones(nFeat,1);
                    alpha(end) = 0; % Do not penalize bias variable
                    funObj = @(w)LogisticLoss(w,Xi,Yi);
                    funObjL2 = @(w)penalizedL2(w,funObj,alpha);
                    wInit = zeros(nFeat,1);
                    w = minFunc(funObjL2,wInit,options);
                    label = sign(X(itestInd{ifold},:)*w*median(Xi(1:8,:)*w));
                    iscore(ifold,l) = mean(label==Y(itestInd{ifold}));
                end
            end
            iscoreAve = mean(iscore);
            [iscoreMax,ind] = max(iscoreAve);
            alphaOpt(fold,sub) = max(alphaRange(iscoreAve==iscoreMax));
            alpha = alphaOpt(fold,sub)*ones(nFeat,1);
%             alpha = alphaRange(ind)*ones(nFeat,1);
            alpha(end) = 0;
            funObj = @(w)logisticLoss(w,X,Y);
            funObjL2 = @(w)penalizedl2(w,funObj,alpha);
            wInit = zeros(nFeat,1);
            w = minFunc(funObjL2,wInit,options);
            label = sign(features(testInd{fold},:)*w*median(X(1:8,:)*w));
            score(fold,sub) = mean(label==labels(testInd{fold}));
        elseif method == 4 % Sparse Logistic Regression
            nGridPts = 6;
            lambdaRange = logspace(-4,1,nGridPts);
            % Internal cross validation for parameter selection            
            iscore = zeros(niFolds,nGridPts);
            for ifold = 1:niFolds
                Xi = X(itrainInd{ifold},:); Yi = Y(itrainInd{ifold});
                for l = 1:nGridPts
                    lambda = lambdaRange(l)*ones(nFeat,1);
                    lambda(end) = 0; % Do not penalize bias variable
                    funObj = @(w)LogisticLoss(w,Xi,Yi);
                    wInit = zeros(nFeat,1);
                    w = L1GeneralProjection(funObj,wInit,lambda,options);
                    label = sign(X(itestInd{ifold},:)*w*median(Xi(1:8,:)*w));
                    iscore(ifold,l) = mean(label==Y(itestInd{ifold}));
                end
            end
            iscoreAve = mean(iscore);
            [iscoreMax,ind] = max(iscoreAve);
            lambdaOpt(fold,sub) = max(lambdaRange(iscoreAve==iscoreMax));
            lambda = lambdaOpt(fold,sub)*ones(nFeat,1);
%             lambda = lambdaRange(ind)*ones(nFeat,1);
            lambda(end) = 0;
            funObj = @(w)logisticLoss(w,X,Y);
            wInit = zeros(nFeat,1);
            w = L1GeneralProjection(funObj,wInit,lambda,options);
            label = sign(features(testInd{fold},:)*w*median(X(1:8,:)*w));
            score(fold,sub) = mean(label==labels(testInd{fold}));
        elseif method == 5 % Logistic Regression with Elastic Net
            nGridPtsAlpha = 5;
            nGridPtsLambda = 6;
            alphaRange = linspace(0.1,0.9,nGridPtsAlpha);
            lambdaRange = logspace(-4,1,nGridPtsLambda);
            % Internal cross validation for parameter selection            
            iscore = zeros(niFolds,nGridPtsAlpha,nGridPtsLambda);
            for ifold = 1:niFolds
                Xi = X(itrainInd{ifold},:); Yi = Y(itrainInd{ifold});
                for a = 1:nGridPtsAlpha
                    for l = 1:nGridPtsLambda
                        alpha = alphaRange(a);
                        lambda = lambdaRange(l)*ones(nFeat,1);
                        lambda(end) = 0; % Do not penalize bias variable
                        funObj = @(w)LogisticLoss(w,Xi,Yi);
                        funObjLap = @(w)penalizedl2(w,funObj,(1-alpha)*lambdaRange(l));
                        wInit = zeros(nFeat,1);                        
                        w = L1GeneralProjection(funObjLap,wInit,alpha*lambda,options);
                        label = sign(X(itestInd{ifold},:)*w*median(Xi(1:8,:)*w));
                        iscore(ifold,a,l) = mean(label==Y(itestInd{ifold}));
                    end
                end
            end
            iscoreAve = squeeze(mean(iscore));
            iscoreMax = max(iscoreAve(:));
            ind = find(iscoreAve==iscoreMax);
            [i,j] = ind2sub([nGridPtsAlpha,nGridPtsLambda],ind);
            jMax = max(j);
            i = max(i(j==jMax));
            j = jMax;
%             [i,j] = ind2sub([nGridPtsAlpha,nGridPtsLambda],ind);
            alphaOpt(fold,sub) = alphaRange(i);
            lambdaOpt(fold,sub) = lambdaRange(j);
            lambda = lambdaOpt(fold,sub)*ones(nFeat,1);
            lambda(end) = 0;
            funObj = @(w)logisticLoss(w,X,Y);
            funObjLap = @(w)penalizedl2(w,funObj,(1-alphaOpt(fold,sub))*lambdaOpt(fold,sub));
            wInit = zeros(nFeat,1);
            w = L1GeneralProjection(funObjLap,wInit,alphaOpt(fold,sub)*lambda,options);
            label = sign(features(testInd{fold},:)*w*median(X(1:8,:)*w));
            score(fold,sub) = mean(label==labels(testInd{fold}));
        elseif method == 6 % Laplacian Sparse Logistic Regression
            nGridPtsAlpha = 5;
            nGridPtsLambda = 6;
            alphaRange = linspace(0.1,0.9,nGridPtsAlpha);
            lambdaRange = logspace(-4,1,nGridPtsLambda);
            % Internal cross validation for parameter selection            
            iscore = zeros(niFolds,nGridPtsAlpha,nGridPtsLambda);
            for ifold = 1:niFolds
                Xi = X(itrainInd{ifold},:); Yi = Y(itrainInd{ifold});
                for a = 1:nGridPtsAlpha
                    for l = 1:nGridPtsLambda
                        alpha = alphaRange(a);
                        lambda = lambdaRange(l)*ones(nFeat,1);
                        lambda(end) = 0; % Do not penalize bias variable
                        funObj = @(w)LogisticLoss(w,Xi,Yi);
                        funObjLap = @(w)LaplacianBN(w,funObj,(1-alpha)*lambdaRange(l),L);
                        wInit = zeros(nFeat,1);                        
                        w = L1GeneralProjection(funObjLap,wInit,alpha*lambda,options);
                        label = sign(X(itestInd{ifold},:)*w*median(Xi(1:8,:)*w));
                        iscore(ifold,a,l) = mean(label==Y(itestInd{ifold}));
                    end
                end
            end
            iscoreAve = squeeze(mean(iscore));
            iscoreMax = max(iscoreAve(:));
            ind = find(iscoreAve==iscoreMax);
            [i,j] = ind2sub([nGridPtsAlpha,nGridPtsLambda],ind);
            jMax = max(j);
            i = max(i(j==jMax));
            j = jMax;            
%             [i,j] = ind2sub([nGridPtsAlpha,nGridPtsLambda],ind);
            alphaOpt(fold,sub) = alphaRange(i);
            lambdaOpt(fold,sub) = lambdaRange(j);
            lambda = lambdaOpt(fold,sub)*ones(nFeat,1);
            lambda(end) = 0;
            funObj = @(w)logisticLoss(w,X,Y);
            funObjLap = @(w)LaplacianBN(w,funObj,(1-alphaOpt(fold,sub))*lambdaOpt(fold,sub),L);
            wInit = zeros(nFeat,1);
            w = L1GeneralProjection(funObjLap,wInit,alphaOpt(fold,sub)*lambda,options);
            label = sign(features(testInd{fold},:)*w*median(X(1:8,:)*w));
            score(fold,sub) = mean(label==labels(testInd{fold}));
        end
    end    
end
% matlabpool('close');

score

