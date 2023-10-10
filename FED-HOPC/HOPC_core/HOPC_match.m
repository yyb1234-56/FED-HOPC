 function corrp = HOPC_match(im_Ref,im_Sen,templateSize,searchRad,cellsize,nCells,orbin,blockoverlay)
%  这个函数为常规匹配模块
%  输入：
%  im_Ref，输入的SAR图像
%  im_Sen，输入的可见光图像
%  templateSize，模板尺寸
%  searchRad，搜索半径
%  cellsize，HOPC的单元尺寸
%  ncells，HOPC的块内单元数
%  orbin，HOPC的计算方向数
%  blockoverlay，特征提取的重叠率
%  输出：
%  corrp，点集，包含兴趣和对应点集


if nargin < 2
    disp('the number input parameters must be >= 2 ');
    return;
end

if nargin < 3
    templateSize = 100;% the template size
end
if nargin < 4
    searchRad = 10;    % the radius of search region
end
if nargin < 5
    cellsize = 4;    % compute the SSD of hopc descriptors
end
if nargin < 6
    nCells = 3;    % compute the SSD of hopc descriptors
end
if nargin < 7
    orbin = 8;    % compute the SSD of hopc descriptors
end
if nargin < 8
    blockoverlay = 0.5;    % compute the SSD of hopc descriptors
end

% tranfer the rgb to gray
[k1,k2,k3] = size(im_Ref);
if k3 == 3
    im_Ref = rgb2gray(im_Ref);
end
im_Ref = double(im_Ref);

[k1,k2,k3] = size(im_Sen);
if k3 == 3
    im_Sen = rgb2gray(im_Sen);
end
im_Sen = double(im_Sen);

[im_RefH,im_RefW] = size(im_Ref);
[im_SenH,im_SenW] = size(im_Sen);

templateRad = round(templateSize/2);    %the template radius
marg=templateRad+searchRad+2;           %the boundary. we don't detect tie points out of the boundary


%extract the interest points using blocked-harris
im1 = im_Ref(marg:im_RefH-marg,marg:im_RefW-marg);% remove the pixel near the boundary
Value = harrisValue(im1);                        % harris intensity value
[r,c,rsubp,cubp] = nonmaxsupptsgrid(Value,3,0.3,5,4); % non-maxima suppression in regular
                                                       % here is 5*5 grid
                                                       % 8 points in each grid, in total 200 interet points 
points1 =[r,c] + marg - 1;
% figure;
% imshow(uint8(im_Sen)); hold on
% plot(points1(:,2),points1(:,1),'g+','MarkerSize',10);


pNum = size(points1,1); % the number of interest points

%caculate the dense block-HOPC descriptor for each pixel

% cellsize=8; % the pixel number (cellsize*cellsize) in a cell
% orbin = 8;  % the number of orientation bin
% nCells = 3;  % the cell number (nCell*nCell) of in a block 
% blockoverlay = 0.5;% the overlay degree between adjacent block
interval =round(cellsize*nCells*(1-blockoverlay));% the pixel interval between adjact block
if interval == 0
    interval =1;
end
%interval =1;
%caculate the dense block-HOPC descriptor 
tic
blockHOPC_Ref = denseBlockHOPC(single(im_Ref),cellsize,orbin,nCells);
blockHOPC_Sen = denseBlockHOPC(single(im_Sen),cellsize,orbin,nCells);
% fprintf('the HOPC descripter time for two images is %fs\n',toc);

%detect the tie points by the template matching strategy for each
C = 0;
for n = 1: pNum
    
    %the x and y coordinates in the reference image
    X_Ref=points1(n,2);
    Y_Ref=points1(n,1);
    
    % get the HOPC descriptor of the template window centered on (X_Ref,Y_Ref)
    HOPC_Ref = single(getDesc(blockHOPC_Ref,Y_Ref,X_Ref,templateRad,interval));

    
    %transform the (x,y) of reference image to sensed image by the geometric relationship of check points 
    %to determine the search region
    tempCo = [X_Ref,Y_Ref];
    
    %tranformed coordinate (X_Sen_c, Y_Sen_c)
    X_Sen_c = tempCo(1);
    Y_Sen_c = tempCo(2);

    %judge whether the transformed points are out the boundary of right image.

    if (X_Sen_c < marg+1 | X_Sen_c > size(im_Sen,2)-marg | Y_Sen_c<marg+1 | Y_Sen_c > size(im_Sen,1)-marg)
        %if out the boundary, this produre enter the next cycle
        continue;
    end
        
    corr = zeros(2*searchRad + 1); % the NCC of HOPC descriptor
     
    % caculate the NCC of HOPC for the search region 
    for i = -searchRad:searchRad
        for j = -searchRad:searchRad
            
            Y_Sen_c2 = Y_Sen_c+i;
            X_Sen_c2 = X_Sen_c+j;
            
            % get the HOPC descriptor of the template window centered on (X_Sen,Y_Sen)
            HOPC_Sen = single(getDesc(blockHOPC_Sen,Y_Sen_c2,X_Sen_c2,templateRad,interval));
           
           %calculate the NCC between two HOPC descriptors 
           temp=corrcoef(HOPC_Ref,HOPC_Sen);
           corr(i + searchRad +1,j + searchRad + 1)=temp(1,2);
        end
    end
    %judge if corr is nan
     nan = isnan(corr);
     if nan(1) > 0
         continue;
     end
    
    %get coordinates with the maximum NCC in the search region 
     maxCorr = max(max(corr));
     max_index = find(corr == maxCorr);
      if(size(max_index,1) > 1)
            % if two maxumal appear, it go to nexe cycle;
         continue;
      end
       
      [max_i,max_j] = ind2sub(size(corr),max_index);
      
      %the (matchY,matchX) coordinates of match
      Y_match = Y_Sen_c-searchRad + max_i-1;
      X_match = X_Sen_c-searchRad + max_j-1;
      C = C+1;
      corrp(C,:)=[X_Ref,Y_Ref,X_match,Y_match];% the coordinates of correct matches
end

end

