% Testing l1General for Sparse Inverse Covariance Estimation
fid = fopen('D:\research\restStateTaskInteg\imagenData\code\subjectList.txt');
nSubs = 65;
for i = 1:nSubs
    sublist{i} = fgetl(fid);
end
nParcel = 1000;
nROIs = 850;
% For extracting upper triangular elements
ind = find(triu(ones(nROIs)));

lambda = [100,75,50,25,10,7.5,5,2.5,1]; % Add 100 back in when finished
nPerm = 500;
for i = 1:length(lambda)
    KrestVec = zeros(length(ind),nSubs); % Store Krest in log space
    % Project precison to log space
    for sub = 1:nSubs
        load(strcat('D:\research\restStateTaskInteg\imagenData\',sublist{sub},'\KrestParcel',int2str(nParcel),'lambda',int2str(lambda(i)*100),'Proj'));
        temp = logm(full(Krest)); clear Krest;
        KrestVec(:,sub) = temp(ind);
    end
    % Compute precision std across subjects
    KrestMean = mean(KrestVec,2);
    KrestStd = sqrt(sum((KrestVec(:)-repmat(KrestMean,nSubs,1)).^2)/(length(ind)*nSubs));
%     save(strcat('KrestStd',int2str(100*lambda)),'KrestStd');
    % Compute permuted precision std across subjects
    KrestStdPerm = zeros(nPerm,1); 
    for k = 1:nPerm
        KrestVecPerm = zeros(length(ind),nSubs);
        for sub = 1:nSubs
            KrestVecPerm(:,sub) = KrestVec(randperm(length(ind)),sub);
        end
        KrestMeanPerm = mean(KrestVecPerm,2);
        KrestStdPerm(k) = sqrt(sum((KrestVecPerm(:)-repmat(KrestMeanPerm,nSubs,1)).^2)/(length(ind)*nSubs));
    end
%     save(strcat('KrestStdPerm',int2str(100*lambda)),'KrestStdPerm');
    (KrestStd-mean(KrestStdPerm))/std(KrestStdPerm)
    lambda
end
