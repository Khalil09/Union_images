% Carregar as imagens.
buildingDir = fullfile('/home/khalil/unb/Computação Visual/union_images', {'1.jpg';'2.jpg'});
buildingScene = imageDatastore(buildingDir);

im1 = readimage(buildingScene, 1);
im2 = readimage(buildingScene, 2);

I1 = rgb2gray(im1);
I2 = rgb2gray(im2);

points1 = detectSURFFeatures(I1);
points2 = detectSURFFeatures(I2);

[features1, points1] = extractFeatures(I1, points1);
[features2, points2] = extractFeatures(I2, points2);

indexPairs = matchFeatures(features1, features2, 'Unique', true, 'MaxRatio', 0.1);
numMatchedPoints = int32(size(indexPairs, 1));

matchedPoints1 = points1(indexPairs(:,1),:);
matchedPoints2 = points2(indexPairs(:,2),:);

tforms2 = projective2d(eye(3));

tforms = estimateGeometricTransform(matchedPoints1, matchedPoints2,...
        'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);

imageSize = size(I1);    

for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(2)], [1 imageSize(1)]);
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
panorama = zeros([height width], 'like', I1);

xLimits = [xMin xMax];
yLimits = [yMin yMax];
    
blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');

panoramaView = imref2d([height width], xLimits, yLimits);    

warpedImage = imwarp(I1, tforms, 'OutputView', panoramaView);
mask = imwarp(true(size(I1,1),size(I1,2)), tforms, 'OutputView', panoramaView);

panorama = step(blender, panorama, warpedImage, mask);

warpedImage2 = imwarp(I2, tforms2, 'OutputView', panoramaView);
mask2 = imwarp(true(size(I2,1),size(I2,2)), tforms2, 'OutputView', panoramaView);

panorama = step(blender, panorama, warpedImage2, mask2);


figure
imshow(panorama)
figure; showMatchedFeatures(I1, I2, matchedPoints1, matchedPoints2);
legend('Image 1', 'Image 2');