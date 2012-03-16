function xout = generic_hdr(TR)
%
% TR is TR interval in seconds
%
% using equation of Glover, creates a generic HDR at intervals TR
%
% Reference:
% Deconvolution of IMpulse Response in Event-Related BOLD fMRI. Neuroimage 9, 416-429 (1999)
%
% martin.mckeown@duke.edu
%

n1 = 5;
t1 = 1.1;  % secs
n2 = 12;
t2 = 0.9;
a2 = 0.4;
scale_val = 1;
delay = 1; 
X = [n1 t1 n2 t2 a2 delay scale_val];
total_time = 15;  %secs
timegrid = 0:TR:total_time;
[xout,timegrid] = hdrfit(X,timegrid);


function [yout,timegrid] = hdrfit(X, timegrid);
n1 = X(1);
t1 = X(2);
n2 = X(3);
t2 = X(4);
a2 = X(5);
delay = X(6);
scale_val = X(7);

maxtime = 20;
if nargin < 2
    timegrid = linspace(0, maxtime);
end
fit_time = timegrid - delay;
ii = find(fit_time < 0);
fit_time(ii) = 0;

c1 = 1/max((fit_time.^n1).*exp(-fit_time/t1));
c2 = 1/max((fit_time.^n2).*exp(-fit_time/t2));

yout = scale_val*(c1*(fit_time.^n1).*exp(-fit_time/t1) - a2 * c2 * (fit_time.^n2) .* exp(-fit_time/t2));


