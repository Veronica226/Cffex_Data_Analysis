filename = ('G:\java\path.txt');
plot_path = 
function plotKpiCurve(filename,plot_path)
    A = importdata(filename);
    m = size(A,1);
    figure(1)
    plot(A(1:m,1),A(1:m,2),'mo')
    xlabel('time');
    ylabel('value');
    hold on
    saveas(gcf,[plot_path,filename,'.png'])
end

    
    
