% filename: string -- don't include iteration number or '.jpeg'
% height: int
% width: int
% borderOn: bool
function plotImages(filename, height, width, borderOn)
close all
%fileFolder should be the path to the ouput image folder
% /Users/izzysalley/tensorflow/proj351/PaintingProcessingEECS351/test_data
fileFolder = fullfile('Users','izzysalley','tensorflow','proj351','PaintingProcessingEECS351','law_quad_layer1');

file = [filename,'0.jpeg'];
filePath = ['/', fullfile(fileFolder, file)];
names{1} = filePath;

modSize = 10;
for k = 1:(width*height)-1
  file = sprintf([filename,'%d.jpeg'], k*modSize);
  filePath = ['/', fullfile(fileFolder, file)];
  names{k+1} = filePath;
end

if borderOn
    saveName = 'test_border';
    sepDist = 3;
    for n = 1:height*width
        img = imread(names{n});
        imArray(:,:,:,n)= img; 
    end
    
    img = createImMontage(imArray,width*height,1,sepDist);
    montage(img);
    
else
    saveName = 'test';
    montage(names, 'Size', [height width]);

end
    fsize = 26;
    ax = gca;
    ax.FontSize = fsize;
    %title([filename, ' Iterations'], 'FontSize',fsize);
    xlabel(sprintf('Optimization Iterations 0 to %d', (k+1)*modSize));
    print(saveName, '-dpng');

end