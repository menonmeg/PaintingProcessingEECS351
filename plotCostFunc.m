% plot cost function
function plotCostFunc(filename)

    fileFolder = ['/', fullfile('Users','izzysalley','tensorflow','proj351','PaintingProcessingEECS351', filename)];
    
    costs = csvread(fileFolder);
    len = numel(costs);
    
    plot(1:len, costs,'k');
    fsize = 24;
    xlabel('Iteration', 'FontSize',fsize);
    ylabel('Cost', 'FontSize',fsize);
    title('Cost Vs. Iteration', 'FontSize',fsize);
    grid on

end
