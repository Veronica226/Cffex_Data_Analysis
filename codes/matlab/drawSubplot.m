close all;
clear;
global file_folder
global plot_dir
file_folder = ('.\subplot_data\');
plot_dir = ('.\kpi-plot\');
event_dir = ('.\cffex-host-alarm\');
event_file_name = strcat(event_dir,'cffex-host-alarm-processed.csv') ;
host_file_name = strcat(event_dir, 'cffex-host-alarm-node-alias.csv');
event_time_list = cell(1,100000);
event_host_list = cell(1,100000);
event_host_id_list = cell(1,100000);
event_level_list = cell(1,100000);
cnt = 0;
fid = fopen(event_file_name, 'r');
while ~feof(fid)
    tline=fgetl(fid);
    if(cnt==0) %first line is header
        cnt = cnt + 1;
        continue;
    end
    tmp_line = regexp(tline, ',', 'split');
    tmp_event_host = tmp_line(1, 2);
    tmp_time = tmp_line(1, 7);  %last time
    tmp_alarm_level = tmp_line(1, 8);
    event_host_id_list(cnt) = tmp_event_host;
    event_time_list(cnt) = tmp_time;
    event_level_list(cnt) = tmp_alarm_level;
    cnt = cnt + 1;
end
fclose(fid);
event_num = cnt - 1;

cnt = 0;
fid = fopen(host_file_name, 'r');
while ~feof(fid)
    tline=fgetl(fid);
    if(cnt==0) %first line is header
        cnt = cnt + 1;
        continue;
    end
    tmp_line = regexp(tline, ',', 'split');
    tmp_host_name = tmp_line(1, 2);
    event_host_list(cnt) = tmp_host_name;
    cnt = cnt + 1;
end
fclose(fid);
host_list_len = cnt - 1;

list = dir([file_folder,'*.csv']);
len = length(list);
for n=1:len
    file_name = strcat (file_folder, list(n).name);
    fprintf('file_name = %s\n', file_name);
    host_name = list(n).name(1:end-4);
    plot_path = strcat (plot_dir,host_name,'\') ;
    if ~exist(plot_path)
        mkdir(plot_path);
    end
    fid1 = fopen(file_name, 'r');
    A = cell(1,100000);  % time_str series
    B = zeros(1, 100000); % max_value array
    C = zeros(1, 100000);% min_value array
    D = zeros(1, 100000);
    E = zeros(1, 100000);
    F = zeros(1, 100000);
    G = zeros(1, 100000);
    H = zeros(1, 100000);
    I = zeros(1, 100000);
    J = zeros(1, 100000);
    K = zeros(1, 100000);
    L = zeros(1, 100000);
    M = zeros(1, 100000);
    N = zeros(1, 100000);
    O = zeros(1, 100000);
    cnt = 0;
    while ~feof(fid1)          %read file
        cnt = cnt + 1;
        tline=fgetl(fid1);
        tmp_line = regexp(tline, ',', 'split');
        tmp_time = tmp_line(1, 1);
        cpu_max = str2double(char(tmp_line(1, 2)));
        cpu_min = str2double(char(tmp_line(1, 3)));
        boot_max = str2double(char(tmp_line(1, 4)));
        boot_min = str2double(char(tmp_line(1, 5)));
        home_max = str2double(char(tmp_line(1, 6)));
        home_min = str2double(char(tmp_line(1, 7)));
        monitor_max = str2double(char(tmp_line(1, 8)));
        monitor_min = str2double(char(tmp_line(1, 9)));
        rt_max = str2double(char(tmp_line(1, 10)));
        rt_min = str2double(char(tmp_line(1, 11)));
        tmp_max = str2double(char(tmp_line(1, 12)));
        tmp_min = str2double(char(tmp_line(1, 13)));
        mem_max = str2double(char(tmp_line(1, 14)));
        mem_min = str2double(char(tmp_line(1, 15)));
        A(cnt) = tmp_time;
        B(1,cnt) = cpu_max;
        C(1,cnt) = cpu_min;
        D(1,cnt) = boot_max;
        E(1,cnt) = boot_min;
        F(1,cnt) = home_max;
        G(1,cnt) = home_min;
        H(1,cnt) = monitor_max;
        I(1,cnt) = monitor_min;
        J(1,cnt) = rt_max;
        K(1,cnt) = rt_min;
        L(1,cnt) = tmp_max;
        M(1,cnt) = tmp_min;
        N(1,cnt) = mem_max;
        O(1,cnt) = mem_min;
    end
    fclose(fid1);
    num = 24 * 30;
    len = floor(cnt / num)-1;
    
    for i = 0:len            %划分时间段
        left_index = i*num+1;
        if(i < len)
            right_index = (i+1)*num;            
        else
            right_index = max((i+1)*num,cnt);
        end
        
        length = right_index-left_index+1;
        startdate = datenum(A(left_index));
        enddate = datenum(A(right_index));
        %fprintf('left=%s, right=%s, length=%d, left=%d, right=%d\n',datestr(startdate),datestr(enddate),(enddate-startdate)*24,left_index,right_index);
      
        h = figure;
        set(gcf,'Position',[0 0 1000 600]);
        set(gcf,'visible', 'off');
        
        sb1 = subplot(5,1,1);
        tile_name = strcat(strrep(host_name, '_', ' '), ' time series plot(',num2str(i+1), ')');
        title(tile_name);
        x = linspace(startdate,enddate,length);
        plot(x,B(1,left_index:right_index),'m-');
        hold on;
        box on;
        plot(x,C(1,left_index:right_index),'b-');
        max_cpu_val = max(max(B(1,left_index:right_index)), max(C(1,left_index:right_index)));
        hold on;
        box on;
        set(gca,'xtick',left_index:1:right_index);
        datetick('x','mm-dd hh');
        %ylim([0 30]);
        ylabel('cpu');
        
        sb2 = subplot(5,1,2);
        plot(x,H(1,left_index:right_index),'m-');
        hold on;
        box on;
        plot(x,I(1,left_index:right_index),'b-');
        max_disk_mn = max(max(H(1,left_index:right_index)), max(I(1,left_index:right_index)));
        hold on;
        box on;
        set(gca,'xtick',left_index:1:right_index);
        datetick('x','mm-dd hh' );
        %ylim([0 50]);
        ylabel('monitor');
        
        sb3 = subplot(5,1,3);
        plot(x,J(1,left_index:right_index),'m-');
        hold on;
        box on;
        plot(x,K(1,left_index:right_index),'b-');
        max_disk_rt = max(max(J(1,left_index:right_index)), max(K(1,left_index:right_index)));
        hold on;
        box on;
        set(gca,'xtick',left_index:1:right_index);
        datetick('x','mm-dd hh' );
        %ylim([0 30]);
        ylabel('rt');
        
        sb4 = subplot(5,1,4);
        plot(x,L(1,left_index:right_index),'m-');
        hold on;
        box on;
        plot(x,M(1,left_index:right_index),'b-');
        max_disk_tmp = max(max(L(1,left_index:right_index)), max(M(1,left_index:right_index)));
        hold on;
        box on;
        set(gca,'xtick',left_index:1:right_index);
        datetick('x','mm-dd hh' );
        %ylim([0 20]);
        ylabel('tmp');
        
        sb5 = subplot(5,1,5);
        plot(x,N(1,left_index:right_index),'m-');
        hold on;
        box on;
        plot(x,O(1,left_index:right_index),'b-');
        max_disk_mem = max(max(N(1,left_index:right_index)), max(O(1,left_index:right_index)));
        hold on;
        box on;
        %ylim([0 50]);
        %xlabel('time');
        ylabel('memory');
        set(gca,'xtick',left_index:1:right_index);
        datetick('x','mm-dd hh' );
        
        for j = 1:event_num
            event_host = event_host_list{str2num(event_host_id_list{j})};
            event_level = str2num(event_level_list{j});
            event_host = lower(event_host);
            if strcmp(host_name ,event_host)==1
                event_time = event_time_list(j);
                tmp_time = event_time{1};
                time = strcat(tmp_time(1:4),'-',tmp_time(5:6),'-',tmp_time(7:end));
                alarm_level = str2num(event_level_list{j});
                event_date = datenum(time);
               % fprintf("count = %d host_name = %s time = %s event_date = %d\n",c,event_host{1},time,event_date);
                if (startdate <= event_date) &&(event_date <= enddate)
                    %text(event_date,6,event_level_list(j));
                    if(event_level == 4)
                        plot(sb1, [event_date,event_date],[0, max_cpu_val+1],'r-', 'linewidth', 1.1);
                        plot(sb2, [event_date,event_date],[0, max_disk_mn+1],'r-', 'linewidth', 1.1);
                        plot(sb3, [event_date,event_date],[0, max_disk_rt+1],'r-', 'linewidth', 1.1);
                        plot(sb4, [event_date,event_date],[0, max_disk_tmp+1],'r-', 'linewidth', 1.1);
                        plot(sb5, [event_date,event_date],[0, max_disk_mem+1],'r-', 'linewidth', 1.1);
                    else
                        plot(sb1, [event_date,event_date],[0, max_cpu_val+1],'g-', 'linewidth', 1.1);
                        plot(sb2, [event_date,event_date],[0, max_disk_mn+1],'g-', 'linewidth', 1.1);
                        plot(sb3, [event_date,event_date],[0, max_disk_rt+1],'g-', 'linewidth', 1.1);
                        plot(sb4, [event_date,event_date],[0, max_disk_tmp+1],'g-', 'linewidth', 1.1);
                        plot(sb5, [event_date,event_date],[0, max_disk_mem+1],'g-', 'linewidth', 1.1);
                    end
                    fprintf("host_name = %s time = %s alarm level = %d event_date = %d\n",event_host,time,alarm_level,event_date);
                end
            end
        end
        
        exportfig(gcf, strcat(plot_path,host_name,'(',num2str(i+1), ')','.png'),'-native', '-m3', '-CMYK')
    end
end




