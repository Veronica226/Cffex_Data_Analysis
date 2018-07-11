close all;
clc;
clear;
global file_folder
global plot_dir
file_folder = ('.\plot-data\');
plot_dir = ('.\kpi-plot\');
list = dir([file_folder,'*.csv']);
len = length(list);
for n=1:len
    file_name = strcat (file_folder, list(n).name) ;
    plot_name = list(n).name(1:end-4);
    %plot_path =  strcat (plot_dir, plot_name,'.png') ;
    %fprintf('file_name = %s\n', file_name);
    draw_plot(file_name,plot_name);
    break;
    %saveas(gcf,[plot_dir,list(n).name(1:end-4),'.png'])
end
function draw_plot(file_name, plot_name)
    global plot_dir;
    host_name_list = regexp(plot_name,'_','split');
    host_name = char(host_name_list(1));
    plot_path = strcat (plot_dir,host_name,'\') ;
    mkdir(plot_path);
    fid = fopen(file_name, 'r');
    A = cell(1,100000);  % time_str series
    B = zeros(1, 100000); % max_value array
    C = zeros(1, 100000); % min_value array
    cnt = 0;
    while ~feof(fid)
        cnt = cnt + 1;
        tline=fgetl(fid);
        tmp_line = regexp(tline, ',', 'split');
        time = tmp_line(1, 1);
        max_value = str2double(char(tmp_line(1, 2)));
        min_value = str2double(char(tmp_line(1, 3)));
        A(cnt) = time;
        B(1,cnt) = max_value;
        C(1,cnt) = min_value;
    end
    formatIn = 'uuuu-MM-dd HH:mm:ss';
    num = 24 * 30;
    len = floor(cnt / num)-1;
    for i = 0:len
        %fprintf('left=%d, right=%d\n',i*num+1,min((i+1)*num,cnt));
        if(i < len)
            right_index = (i+1)*num;
        else
            right_index = max((i+1)*num,cnt);
            %x = datetime(A(i*num+1:max((i+1)*num,cnt)), 'InputFormat', formatIn);
        end
        x = datetime(A(i*num+1:right_index), 'InputFormat', formatIn);

        startdate = datenum(A(i*num+1));
        enddate = datenum(A(min((i+1)*num,cnt)));
        fprintf('left=%s, right=%s, length=%d, left=%d, right=%d\n',datestr(startdate),datestr(enddate),(enddate-startdate)*23,i*num+1,right_index);
        %t = linspace(startdate,enddate,720);
        %plot(t,B(1,i*num+1:right_index),'b-');
        %datetick('t',15);



        figure;
        plot(x,B(1,i*num+1:right_index),'m-');
        %plot(x,B(1,i*num+1:min((i+1)*num,cnt)),'m-');
        xlabel('time');
        ylabel('value');
        %axis ( [2018012400 2018061123 0 20] )
        hold on;
        box on;
        plot(x,C(1,i*num+1:right_index),'b-');
        %plot(x,C(1,i*num+1:min((i+1)*num,cnt )),'b-');
        h = legend('maxvalue','minvalue');
        set(h, 'Location','NorthWest');
        tile_name = strcat(strrep(plot_name, '_', ' '), ' time series plot(',num2str(i+1), ')');
        title(tile_name);
        exportfig(gcf, strcat(plot_path,plot_name,'(',num2str(i+1), ')','.png'),'-native', '-transparent', '-m3', '-CMYK')
        %saveas(gcf, strcat(plot_path,plot_name,'(',num2str(i+1), ')','.png'));
    end
end


    
    
