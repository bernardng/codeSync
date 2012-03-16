% Faces classification 
clear all;
close all;

localPath = '/volatile/bernardng/data/imagen/';
netwPath = '/home/bn228083/';
addpath(genpath([netwPath,'MatlabToolboxes/markSchmidtCode']));
addpath(genpath([netwPath,'MatlabToolboxes/general']));
addpath(genpath([netwPath,'programs/l1_logreg-0.8.2/']));

fid = fopen([localPath,'subjectLists/facesList.txt']);
nSubs = 58;
sublist = cell(nSubs,1);
for i = 1:nSubs
    sublist{i} = fgetl(fid);
end

nFolds = 5;
niFolds = 4;
options.Method = 'lbfgs'; 
options.optTol = 1e-9; 
options.progTol = 1e-9; 
nGridPts = 5;
% C = linspace(1e0,1e9,nGridPts);
C = logspace(-3,3,nGridPts);
method = 1;
subs = 1:5;

score = zeros(nFolds,length(subs));
for sub = subs
    disp(['Subject ',int2str(sub)]);
    [features,labels] = loadFeatLabel(sublist{sub});
%     load([localPath,sublist{sub},'/restfMRI/K_rest_anatDense_parcel500_quic335_cv.mat']);
    load([localPath,sublist{sub},'/dwi/K_anatDense_parcel500.mat']);
    K = diag(sum(Kanat))-Kanat;
    
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
%             cvx_begin
%                 variables w(nFeat)
%                 minimize mean(log(1+exp(-Y.*(X*w))))+1e-50*sum(w(1:end-1).^2)
%             cvx_end
            label = sign(features(testInd{fold},:)*w);
            score(fold,sub) = mean(label==labels(testInd{fold}));
        elseif method == 2 % Support Vector Machine
        elseif method == 3 % l2-regularized Logistic Regression
%             % Internal cross validation for parameter selection            
%             iscore = zeros(niFolds,nGridPts);
%             for ifold = 1:niFolds
%                 XX = X(itrainInd{ifold},:); YY = Y(itrainInd{ifold});
% %                 lambdaRange = size(XX,1)./C;
%                 lambdaRange = 1./C;
%                 for l = 1:nGridPts
%                     lambda = lambdaRange(l)*ones(nFeat,1);
%                     lambda(end) = 0; % Do not penalize bias variable
% %                     funObj = @(w)LogisticLoss(w,XX,YY);
% %                     funObjL2 = @(w)penalizedL2(w,funObj,lambda);
% %                     wInit = zeros(nFeat,1);
% %                     w = minFunc(funObjL2,wInit,options);
%                     cvx_begin
%                         variables w(nFeat)
%                         minimize mean(log(1+exp(-YY.*(XX*w))))+lambdaRange(l)*sum(w(1:end-1).^2)
%                     cvx_end
%                     label = sign(X(itestInd{ifold},:)*w);
%                     iscore(ifold,l) = mean(label==Y(itestInd{ifold}));
%                 end
%             end
%             [dummy,ind] = max(mean(iscore));
% %             lambdaOpt = size(X,1)./C(ind);
%             lambdaOpt = 1./C(ind)
%             lambda = lambdaOpt*ones(nFeat,1);
%             lambda(end) = 0;
%             funObj = @(w)logisticLoss(w,X,Y);
%             funObjL2 = @(w)penalizedl2(w,funObj,lambda);
%             wInit = zeros(nFeat,1);
%             w = minFunc(funObjL2,wInit,options);
%             cvx_begin
%                 variables w(nFeat)
%                 minimize mean(log(1+exp(-Y.*(X*w))))+lambdaOpt*sum(w(1:end-1).^2)
%             cvx_end
%             w = slrIntPt(X,Y,1e-1,100,1e-6);
            label = sign(features(testInd{fold},:)*w*median(features(trainInd{fold}(1:8),:)*w));
            score(fold,sub) = mean(label==labels(testInd{fold}));
        elseif method == 4 % Sparse Logistic Regression
            
        elseif method == 5 % Laplacian Logistic Regression
        end
    end    
end
score

