close all;
clc;
clear;
global file_folder
file_folder = ('.\cluster_data\');
file_name = strcat(file_folder,'tsne.csv');
fid = fopen(file_name, 'r');

A = zeros(1,100000);  
B = zeros(1, 100000); 
C = zeros(1, 100000);
D = zeros(1, 100000);

while ~feof(fid)
        cnt = cnt + 1;
        tline=fgetl(fid);
        tmp_line = regexp(tline, ',', 'split');
        x = str2double(char(tmp_line(1, 1)));
        y = str2double(char(tmp_line(1, 2)));
        z = str2double(char(tmp_line(1, 3)));
        c = str2double(char(tmp_line(1, 4)));
        A(1,cnt) = x;
        B(1,cnt) = y;
        C(1,cnt) = z;
        D(1,cnt) = c;
end

plot(A,B,C)



