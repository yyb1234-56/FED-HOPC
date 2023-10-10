 function corrp = WED_HOPC_match(im_Ref,im_Sen,templateSize,searchRad,num_p,gamma1,gamma2,cellsize,nCells,orbin,blockoverlay)
%  这个函数为非均匀纹理场景匹配优化模块
%  输入：
%  im_Ref，输入的SAR图像
%  im_Sen，输入的可见光图像
%  templateSize，模板尺寸
%  searchRad，搜索半径
%  num_p，保留兴趣点数量
%  gamma1，WED1的阈值
%  gamma2，WED2的阈值
%  cellsize，HOPC的单元尺寸
%  ncells，HOPC的块内单元数
%  orbin，HOPC的计算方向数
%  blockoverlay，特征提取的重叠率
%  输出：
%  corrp，点集，包含兴趣和对应点集

 addpath WED
 
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
    num_p = 50;
end
if nargin < 6
    gamma1 = 0.15;    % the radius of search region
end
if nargin < 7
    gamma2 = 0.15;    % the radius of search region
end
if nargin < 8
    cellsize = 4;    % compute the SSD of hopc descriptors
end
if nargin < 9
    nCells = 3;    % compute the SSD of hopc descriptors
end
if nargin < 10
    orbin = 8;    % compute the SSD of hopc descriptors
end
if nargin < 11
    blockoverlay = 0.5;    % compute the SSD of hopc descriptors
end

% tranfer the rgb to gray
[~,~,k3] = size(im_Ref);
if k3 == 3
    im_Ref = rgb2gray(im_Ref);
end
im_Ref = double(im_Ref);

[~,~,k3] = size(im_Sen);
if k3 == 3
    im_Sen = rgb2gray(im_Sen);
end
im_Sen = double(im_Sen);

[im_RefH,im_RefW] = size(im_Ref);

templateRad = round(templateSize/2);    %the template radius
marg=templateRad+searchRad+2;           %the boundary. we don't detect tie points out of the boundary

%   计算WED
[M,~,~,~,~,~,~] = phasecong3(im_Ref,4,6,3,'mult',1.6,'sigmaOnf',0.75,'g', 3, 'k',1);
WED = Texture_density(M,21,10);
a=max(WED(:)); b=min(WED(:)); WED=(WED-b)/(a-b);
MASK = imbinarize(WED, gamma2);
se = strel('square',10);    
MASK = imopen(MASK,se);  %开操作

%   WED1
WED = WED(marg:im_RefH-marg,marg:im_RefW-marg);
[r,c] = nonmaxsupptsgrid_ED(WED,3,gamma1,20,1);
points = [];
for i=1:size(r,1)
    points = [points;c(i),r(i),WED(r(i),c(i))];
end
sort_points = sortrows(points,[3,1],'descend'); 
if size(sort_points,1)>num_p
    init_CP_Ref = sort_points(1:num_p,1:2);
else
    init_CP_Ref = sort_points(:,1:2);
end
init_CP_Ref = init_CP_Ref + marg - 1;
pNum = size(init_CP_Ref,1); % the number of interest points

%caculate the dense block-HOPC descriptor for each pixel
interval =round(cellsize*nCells*(1-blockoverlay));% the pixel interval between adjact block
if interval == 0
    interval =1;
end

%caculate the dense block-HOPC descriptor 
blockHOPC_Ref = denseBlockHOPC(single(im_Ref),cellsize,orbin,nCells);
blockHOPC_Sen = denseBlockHOPC(single(im_Sen),cellsize,orbin,nCells);
% fprintf('the HOPC descripter time for two images is %fs\n',toc);
size_Ref = size(blockHOPC_Ref);
size_Sen = size(blockHOPC_Sen);

%detect the tie points by the template matching strategy for each
C = 0;
for n = 1: pNum
    %     disp(n)
    %the x and y coordinates in the reference image
    X_Ref=init_CP_Ref(n,1);
    Y_Ref=init_CP_Ref(n,2);

    %   WED2       
    calc_points = need_calc(MASK,X_Ref-templateRad,Y_Ref-templateRad,templateSize,interval,cellsize*nCells);
    N = size(calc_points,1);
    if N==0 % 如果MASK全为零，则不计算该点的匹配结果
        continue;
    end
    % get the HOPC descriptor of the template window centered on (X_Ref,Y_Ref)
    HOPC_Ref = single(ED2_getDesc(blockHOPC_Ref,calc_points,size_Ref,N));

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
            calc_points2 = calc_points + [Y_Sen_c2-Y_Ref,X_Sen_c2-X_Ref];

            % get the HOPC descriptor of the template window centered on (X_Sen,Y_Sen)
            HOPC_Sen = single(ED2_getDesc(blockHOPC_Sen,calc_points2,size_Sen,N));

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

