import os
import numpy as np
from scipy.io import loadmat

data = loadmat("data/hipp_2dtrack_a/smJun03p2.dat")

N = 49

data = reshape(data, 3, length(data)/3);
data = data';
size(data)  %  43799-by-3

fclose(fid);

% sampling time
Ts = 0.0333;
duration  = size(data,1) * Ts;  % in second


Tmax = data(end, 3);
Tmin = data(1,3);

time_edges = [Tmin: 0.25: Tmax];  %  250 ms per bin

% interpolated rat's position in time bins
Rat_pos = interp1(data(:, 3), [data(:, 1), data(:, 2)], time_edges');

vel = abs(diff(Rat_pos, 1, 1 )); % row difference
vel = [vel(1, :); vel];
% 250 ms
rat_vel = 4 * sqrt(vel(:, 1).^2 + vel(:, 2).^2);  % unit: cm/s
vel_ind = find(rat_vel >= 10);  % RUN velocity threshold

% using RUN only
T = length(vel_ind);
% using Run + pause periods
T = length(time_edges);

AllSpikeData = zeros(C,T);

for i=1:C
      str = ['Cell_num' num2str(i)];
      fid = fopen(str, 'r');

     cell_data = fscanf(fid, '%f');
    cell_data = reshape(cell_data, 3, length(cell_data)/3)';
    spike_time = cell_data(:, 3);
    spike_pos = cell_data(:, 1:2);

    [spike_time_count, bin] = histc(spike_time, time_edges);   % column vector

    % if analyzing the RUN period only uncomment this
    % spike_time_count = spike_time_count(vel_ind);


    AllSpikeData(i, :) = spike_time_count';
    fclose(fid);

end