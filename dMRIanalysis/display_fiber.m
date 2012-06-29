function H = displayFiber(tracts,selection,colorStyle,showStartEnd,realXYZ,lineStyle)
% ------------------------------------------------------
%   H = tracTubes_DTIstudio_selection(tracts,selection,colorStyle,showStartEnd,realXYZ,lineStyle);
% ------------------------------------------------------
% 
% A function to plot the lines imported from DTI studio (or other
% programs). 
% 
%   tracts       : The tract structure. To read it into matlab, use
%                  f_readFiber.m (for DTIstudio) or f_readFiber_vtk_bin
%                  (for MedINRIA).
%   selection    : A vector of line indices
%   colorStyle   : Use 'origin' to color each line according to the xyz
%                  coordinate of its first vertex. Anything else will use the random colors
%                  assigned by DTIstudio. Default is the random color.
%                  If colorStyle is a 1x3 vector, it is taken as an rgb
%                  triple and the tracts will be colored according to this value.
%   showStartEnd : 1 or 0
%   realXYZ      : Use world coordinates, as defined within the tract. The
%                  default is voxel coordinates.
%   lineStyle    : Matlab's way of line styles, for example '-ob'.
%
%
%
% Luis Concha. MNI. January, 2008. Based on a script made at the University
% of Alberta (tracTubes).
%
%   See also f_readFiber, f_readFiber_vtk_bin

if nargin < 3
    reduction = 1;
    colorStyle = 'random';
    showStartEnd = 1;
    realXYZ = 0;
    lineStyle = '-';
elseif nargin<4
    colorStyle = 'random';
    showStartEnd = 1;
    realXYZ = 0;
    lineStyle = '-';
elseif nargin<5
    showStartEnd = 1;
    lineStyle = '-';
elseif nargin<6
    realXYZ = 0;
    lineStyle = '-';
elseif nargin<6
    lineStyle = '-';
end

firstPoints = zeros(length(selection),3);
lastPoints  = zeros(length(selection),3);


for lineidx = 1 : length(selection)
   thisLine = selection(lineidx);
   myLine = tracts.fiber(thisLine).xyzFiberCoord;
   if realXYZ
      myLine = myLine .* repmat([tracts.fPixelSizeWidth tracts.fPixelSizeHeight tracts.fSliceThickness],...
                                length(myLine),1);
   end
   myLineData = sum(tracts.fiber(thisLine).rgbFiberColor); 
   
   if strcmp(colorStyle,'origin')
        theColor(1) = tracts.fiber(thisLine).xyzFiberCoord(1,1)./tracts.nImgWidth;
        theColor(2) = tracts.fiber(thisLine).xyzFiberCoord(1,2)./tracts.nImgHeight;
        theColor(3) = tracts.fiber(thisLine).xyzFiberCoord(1,3)./tracts.nImgSlices;
   elseif ~ischar(colorStyle) && length(colorStyle) == 3
        theColor = colorStyle;
   else
       theColor = tracts.fiber(thisLine).rgbFiberColor;
       if max(theColor) > 1
          theColor = theColor ./ 255; 
       end
   end
   H.lines(thisLine)=plot3(myLine(:,1),myLine(:,2),myLine(:,3),lineStyle);%'Color',theColor,'LineStyle',lineStyle);
   set(H.lines(thisLine),'Color',theColor);
   hold on
   
   % Save the first and last coordinate;
   firstPoints(lineidx,:) = myLine(1,:);
   lastPoints(lineidx,:)  = myLine(size(myLine,1),:);
   
end

set(gcf,'Renderer','OpenGL');

index = find(H.lines==0);
H.lines(index)=[];

% now plot the first points as dots
if showStartEnd == 1
   H.start = plot3(firstPoints(:,1),firstPoints(:,2),firstPoints(:,3),'og');
   H.stop  = plot3(lastPoints(:,1),lastPoints(:,2),lastPoints(:,3),'xr');
   hold off
end
