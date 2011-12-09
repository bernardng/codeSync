% Generates random precision matrix
Ktemp = spdiags(ones(nROIs,3),-1:1,nROIs,nROIs);
Ktemp(1:nActive,1:nActive) = 1;
Kint = sprandsym(Ktemp,[],0.75,3);
indDiag = find(eye(nROIs));
ind = find(Kint);
indOffDiag = setdiff(ind,indDiag);
pcorrMax = max(Kint(indOffDiag));
Kint(ind) = Kint(ind)-pcorrMax;
Kint(indDiag) = 2*Kint(indDiag);