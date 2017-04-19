%% subplot images
% example: subplotImages({'individual_channel_norm.png','all_channel_norm.png','vgg_mean.png',},3,{'Individual Normalization','Uniform Normalization','VGG Mean'},3,1, 'image_init_combo');
% example2: subplotImages({'sunset_lake_cropped.jpg','picasso_cubism_cropped.jpg','van_gough_cafe_cropped.jpg','kandinsky_cropped.jpg',},4,{'','','',''},2,2, 'style_dict');
function subplotImages(imNames, numImg, labels, totRow, totCol, saveName)

for i = 1:numImg
   img = imread(imNames{i});
   subplot(totRow, totCol,i);
%    subaxis(totRow,totCol,i, 'Spacing', 0.03, 'Padding', 0, 'Margin', 0);
   imshow(img);
   title(labels{i}, 'FontSize',26);
end

set(gca, 'LooseInset', get(gca,'TightInset')) % make subplots closer together 

% save final image
print(saveName, '-dpng');

end