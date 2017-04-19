% filename: string -- don't include iteration number or '.jpeg'
% saveName: string
% numImg: int
% borderOn: bool
% height: int
% width: int
% example: plotImages('','vgg_mean_lion',6,1, 1,1)
function plotImages(filename, saveName, numImg, borderOn, height, width)
%close all
%fileFolder should be the path to the ouput image folder
% /Users/izzysalley/tensorflow/proj351/PaintingProcessingEECS351/test_data
fileFolder = fullfile('Users','izzysalley','tensorflow','proj351','ImageSave', 'vgg_mean_lion'); %'PaintingProcessingEECS351'

firstInd = '50';
file = [filename,firstInd,'.jpeg'];
filePath = ['/', fullfile(fileFolder, file)];
names{1} = filePath;

modSize = 100;
for k = 1:numImg-1
  file = sprintf([filename,'%d.jpeg'], k*modSize);
  filePath = ['/', fullfile(fileFolder, file)];
  names{k+1} = filePath;
end

if borderOn
    sepDist = 3;
    for n = 1:numImg
        img = imread(names{n});
        imArray(:,:,:,n)= img; 
    end
    
    img = createImMontage(imArray,numImg,1,sepDist);
    montage(img);
    
else
    montage(names, 'Size', [height width]);

end
    fsize = 26;
    ax = gca;
    ax.FontSize = fsize;
    %title([filename, ' Iterations'], 'FontSize',fsize);
    xlabel(sprintf('Optimization Iterations 50 to %d', k*modSize));
    %xlabel(sprintf('Optimization Iterations 50, 100, 200, 300, 400, 500'));
    print(saveName, '-dpng');

end