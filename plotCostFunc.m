% plot cost function
function plotCostFunc(filename)

    costs = csvread(filename);
    len = numel(filename);
    
    plot(1:len, costs,'k');
    fsize = 24;
    xlabel('Iteration', 'FontSize',fsize);
    ylabel('Cost', 'FontSize',fsize);
    title('Cost Vs. Iteration', 'FontSize',fsize);
    grid on

end