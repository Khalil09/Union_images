% Carregar as imagens.
buildingDir = fullfile('/home', 'khalil/unb/Computação Visual/union_images', {'1.jpg';
                                                                              '2.jpg';
                                                                              '3.jpg';
                                                                              '4.jpg'
                                                                              });
buildingScene = imageDatastore(buildingDir);

im = readimage(buildingScene, 1);

I = rgb2gray(im);
points = detectSURFFeatures(I);
[features, points] = extractFeatures(I, points);

numImages = numel(buildingScene.Files);
tforms(numImages) = projective2d(eye(3));

for n = 2:numImages
    
    pointsPrev = points;
    featuresPrev = features;
    
    im = readimage(buildingScene, n);
    I = rgb2gray(im);
    points = detectSURFFeatures(I);
    [features, points] = extractFeatures(I, points);
    
    indexPairs = matchFeatures(features, featuresPrev, 'Unique', true);
    matchedPoints = points(indexPairs(:,1),:);
    matchedPointsPrev = pointsPrev(indexPairs(:,2),:);
    
    tforms(n) = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
        'projective', 'Confidence', 99.9, 'MaxNumTrials', 4000);
   
    tforms(n).T = tforms(n).T * tforms(n-1).T;
   
end

imageSize = size(I);    

for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(2)], [1 imageSize(1)]);
end

avgXLim = mean(xlim, 2);

[~, idx] = sort(avgXLim);

centerIdx = floor((numel(tforms)+1)/2);

centerImageIdx = idx(centerIdx);

Tinv = invert(tforms(centerImageIdx));

for i = 1:numel(tforms)
    tforms(i).T = tforms(i).T * Tinv.T;
end

% Find the minimum and maximum output limits
xMin = min([1; xlim(:)]);
xMax = max([imageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([imageSize(1); ylim(:)]);

% Width and height of panorama.
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Initialize the "empty" panorama.
panorama = zeros([height width 3], 'like', I);

xLimits = [xMin xMax];
yLimits = [yMin yMax];
    
blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');

panoramaView = imref2d([height width], xLimits, yLimits);    


for i = 1:numImages
    
    I = readimage(buildingScene, i);
     
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
    
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);
    panorama = step(blender, panorama, warpedImage, mask);
end

figure
imshow(panorama)