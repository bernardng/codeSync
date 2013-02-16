% Display ROC
filepath = 'I:\research\code\imagen\groupConnectivityInference\sig\sig';
template = '59subs_ica_roi_parcel500_refined';

% Subject level plot
% OLS
load([filepath,'OLS',template]);
nDetOLS = squeeze(mean(mean(sig)));
% OAS 
load([filepath,'OAS',template]);
nDetOAS = squeeze(mean(mean(sig)));
% SGGM
load([filepath,'SGGM',template]);
nDetSGGM = squeeze(mean(mean(sig)));
% SGGGM
load([filepath,'KsgggmADMMsub',template]);
nDetSGGGMsubbias = squeeze(mean(mean(sig)));
load([filepath,'KsgggmADMMsubRand',template]);
nDetSGGGMsubrand = squeeze(max(mean(mean(sig)),[],4));
dnDet = nDetSGGGMsubrand-nDetOLS;
nDetSGGGMsubunbias = nDetSGGGMsubbias-dnDet;

% figure;
pThresh = 0:0.0025:0.0475;
plot(pThresh,flipud(nDetSGGGMsubunbias(end-19:end)),'r');
hold on; plot(pThresh,flipud(nDetSGGM(end-19:end)),'b');
hold on; plot(pThresh,flipud(nDetOAS(end-19:end)),'g');
hold on; plot(pThresh,flipud(nDetOLS(end-19:end)),'k');
% hold on; plot(pThresh,flipud(nDetSGGGMrand(end-19:end)),'m');

% Group level plot
% OAS Euclidean
load([filepath,'KoasEuclSubMean',template]);
nDetOASeuclbias = squeeze(mean(mean(sig)));
load([filepath,'KoasEuclSubMeanRand',template]);
nDetOASeuclrand = squeeze(max(mean(mean(sig)),[],4));
dnDet = nDetOASeuclrand-nDetOLS;
nDetOASeuclunbias = nDetOASeuclbias-dnDet;
% OAS Log Euclidean
load([filepath,'KoasLogEuclSubMean',template]);
nDetOASlogeuclbias = squeeze(mean(mean(sig)));
load([filepath,'KoasLogEuclSubMeanRand',template]);
nDetOASlogeuclrand = squeeze(max(mean(mean(sig)),[],4));
dnDet = nDetOASlogeuclrand-nDetOLS;
nDetOASlogeuclunbias = nDetOASlogeuclbias-dnDet;
% OAS Log Euclidean
load([filepath,'KoasConcatAll',template]);
nDetOASconcatallbias = squeeze(mean(mean(sig)));
load([filepath,'KoasConcatAllRand',template]);
nDetOASconcatallrand = squeeze(max(mean(mean(sig)),[],4));
dnDet = nDetOASconcatallrand-nDetOLS;
nDetOASconcatallunbias = nDetOASconcatallbias-dnDet;
% SGGM Euclidean
load([filepath,'KsggmEuclSubMean',template]);
nDetSGGMeuclbias = squeeze(mean(mean(sig)));
load([filepath,'KsggmEuclSubMeanRand',template]);
nDetSGGMeuclrand = squeeze(max(mean(mean(sig)),[],4));
dnDet = nDetSGGMeuclrand-nDetOLS;
nDetSGGMeuclunbias = nDetSGGMeuclbias-dnDet;
% SGGM Log Euclidean
load([filepath,'KsggmLogEuclSubMean',template]);
nDetSGGMlogeuclbias = squeeze(mean(mean(sig)));
load([filepath,'KsggmLogEuclSubMeanRand',template]);
nDetSGGMlogeuclrand = squeeze(max(mean(mean(sig)),[],4));
dnDet = nDetSGGMlogeuclrand-nDetOLS;
nDetSGGMlogeuclunbias = nDetSGGMlogeuclbias-dnDet;
% SGGM Log Euclidean
load([filepath,'KsggmConcatAll',template]);
nDetSGGMconcatallbias = squeeze(mean(mean(sig)));
load([filepath,'KsggmConcatAllRand',template]);
nDetSGGMconcatallrand = squeeze(max(mean(mean(sig)),[],4));
dnDet = nDetSGGMconcatallrand-nDetOLS;
nDetSGGMconcatallunbias = nDetSGGMconcatallbias-dnDet;
% SGGGM 
load([filepath,'KsgggmADMMgrp',template]);
nDetSGGGMgrpbias = squeeze(mean(mean(sig)));
load([filepath,'KsgggmADMMgrpRand',template]);
nDetSGGGMgrprand = squeeze(max(mean(mean(sig)),[],4));
dnDet = nDetSGGGMgrprand-nDetOLS;
nDetSGGGMgrpunbias = nDetSGGGMgrpbias-dnDet;

% figure;
pThresh = 0:0.0025:0.0475;
hold on; plot(pThresh,flipud(nDetSGGGMgrpunbias(end-19:end)),'r');
hold on; plot(pThresh,flipud(nDetSGGMeuclunbias(end-19:end)),'b:');
hold on; plot(pThresh,flipud(nDetSGGMlogeuclunbias(end-19:end)),'b--');
hold on; plot(pThresh,flipud(nDetSGGMconcatallunbias(end-19:end)),'b');
hold on; plot(pThresh,flipud(nDetOASeuclunbias(end-19:end)),'g:');
hold on; plot(pThresh,flipud(nDetOASlogeuclunbias(end-19:end)),'g--');
hold on; plot(pThresh,flipud(nDetOASconcatallunbias(end-19:end)),'g');
hold on; plot(pThresh,flipud(nDetOLS(end-19:end)),'k');
