%%
close all;
clear all;
clc;

results = [
    jsondecode(fileread('spice.json'));
    jsondecode(fileread('bsim.json'));
    jsondecode(fileread('neurongpu.json'))
];

C = distinct_colors(20, 'w');
global CM;
CM = [C(13,:); C(8,:); C(5,:)];

global TFS; global LFS;
TFS = 16;
LFS = 14;

%% Sim. time as function of network size (single GPU)
models = {'vogels' 'brunel' 'brunel+' 'synth_0.00156_0.005_1'};
titles = {'Vogels' 'Brunel' 'Brunel+' 'Synthetic'};
for i = 1:4
    plot_group(filter(results, 'simtime', {'x_gpus' 1 'model' models{i}}, 'sim'));
    title(titles{i}, "FontSize", TFS);
    if i == 4
        xlabel({'Synapse Count'; 'Neuron Count'}, 'FontSize', LFS);
        %set(gca, 'YScale', 'log');
    else
        xlabel('Synapse Count', 'FontSize', LFS);
    end
    ylabel('Simulation Time (s)', 'FontSize', LFS);
    if strcmp(models{i}, 'brunel+')
        xlim([0 7.5e8]);
        xticks([0:0.25:0.75] * 1e9);
        xticklabels({'0' '0.25B' '0.5B' '0.75B'});
    else
        if i == 4
            xticks([0:0.5:3] .* 1e9);
            xticklabels({'0\newline0' '0.5B\newline0.6M' '  1B\newline0.8M' '1.5B\newline 1M' '  2B\newline1.1M' '2.5B\newline1.3M' '  3B\newline1.4M'});
        else
            xticks([0:0.5:3] .* 1e9);
            xticklabels({'0' '0.5B' '1B' '1.5B' '2B' '2.5B' '3B'});
        end
    end
	plot([0 3] .* 1e9, [10 10], 'Color', 'r', 'LineStyle', '--', 'HandleVisibility', 'off');
    
    saveas(gcf, strcat('simtime_', models{i}, '.eps'), 'epsc');
end

%% Setup time as a function of network size
figure;
hold on;
sims = {'BSim' 'NeuronGPU' 'Spice'};
ngpus = [1 2 4 8];
markers = {'-' '--' '-.' ':'};
legends = {};

for i = 1:length(sims)
    for j = 1:length(ngpus)
        xy = filter_unique(results, 'setuptime', {'sim' sims{i} 'x_gpus' ngpus(j) 'model' 'synth_0.05_0.005_1'});
        if length(xy) == 0
            continue;
        end
        plot(xy(1,:), xy(2,:), 'LineWidth', 2, 'Color', CM(i,:), 'LineStyle', markers{j}, 'MarkerSize', 8);
        if compare('NeuronGPU', sims{i})
            legends{end+1} = sims{i};
        else
            legends{end+1} = strcat(sims{i}, ' (', num2str(ngpus(j)), ' GPUs)');
        end
    end
end
title('Setup', 'FontSize', TFS);
xlabel('Synapse Count', 'FontSize', LFS);
xlim([2e8 2.4e10]);
xticks([2e8 1e9 1e10 2.4e10]);
xticklabels({'0.2B' '1B' '10B' '24B'});

ylabel('Setup Time (s)', 'FontSize', LFS);
ylim([0 1e3]);
yticks([1 10 100 1e3]);
yticklabels({'1' '10' '100' '1000'});

legend(legends, 'FontSize', LFS);
grid on;
set(gca, 'YScale', 'log');
set(gca, 'XScale', 'log');
tmp = get(gca, 'XTickLabel');  
set(gca, 'XTickLabel', tmp, 'fontsize', LFS);

saveas(gcf, 'setuptime.eps', 'epsc');

%% Sim. time as a function of network size (multi-GPU)
old = CM;
CM = [C(11,:); C(9,:); C(13,:); C(18,:)];
for model = {'Vogels' 'Brunel' 'Brunel+'}
    plot_group(filter(results, 'simtime', {'sim' 'Spice' 'model' model{1}}, 'x_gpus'));
    title(model);
end

for model = {'Vogels' 'Brunel'}
    plot_group(filter(results, 'simtime', {'sim' 'BSim' 'model' model{1}}, 'x_gpus'));
    title(model);
end
CM = old;

%% Speedup & Scaleup
old = CM;
CM = [C(6,:); C(20,:); C(18,:)];
plot_scale(results, 'Spice', {'Vogels' 'Brunel' 'Brunel+'});
plot_scale(results, 'BSim', {'Vogels' 'Brunel'});
CM = old;

function plot_scale(results, sim, models)
    global CM;
    global TFS; global LFS;
    
    speedup = ones(1, length(models));
    scaleup = speedup;
    for igpu = 1:3
        s1 = [];
        s2 = [];

        for model = models
            xy1 = filter_unique(results, 'simtime', {'sim' sim 'x_gpus' 1 'model' model{1}});
            xy2 = filter_unique(results, 'simtime', {'sim' sim 'x_gpus' 2^igpu 'model' model{1}});

            s1 = [s1 xy1(2,end) / lerp(xy2(2,:), indexof(xy1(1,end), xy2(1,:)))];

            s2 = [s2 xy1(2,end) / lerp(xy2(2,:), indexof(xy1(2,end), xy2(2,:))) *...
                lerp(xy2(1,:), indexof(xy1(2,end), xy2(2,:))) / xy1(1,end)];
        end

        speedup = [speedup; s1];
        scaleup = [scaleup; s2];
    end
    
    data = {speedup scaleup};
    ylabels = {'Speedup (x)' 'Scaleup (x)'};
    ymax = {6, 9};
    for i = 1:2
        figure;
        hold on;
        
        xlim([0.5 4.5]);
        ylim([0 ymax{i}]);
        map = get(gca, 'ColorOrder');
        for e = 1:floor(log2(ymax{i}))
            plot([0.5 4.5], [2^e 2^e], 'HandleVisibility', 'off', 'LineWidth', 0.5, 'Color', 0.5*ones(1,3));
        end
        set(gca, 'ColorOrder', CM, 'NextPlot','ReplaceChildren')
        
        bar(data{i});
        
        title(sim, 'FontSize', TFS);
        xlabel('GPU Count', 'FontSize', LFS);
        xticklabels({'1', '2', '4', '8'});
        ylabel(ylabels{i}, 'FontSize', LFS);
        yticks([0:ymax{i}]);
        legend(models, 'Location', 'Northwest', 'FontSize', LFS);
        set(gca, 'YGrid', 'on');
        
        tmp = get(gca, 'XTickLabel');  
        set(gca, 'XTickLabel', tmp, 'fontsize', LFS);
        
        if i==1
            saveas(gcf, strcat('speedup', sim, '.eps'), 'epsc');
        else
            saveas(gcf, strcat('scaleup', sim, '.eps'), 'epsc');
        end
    end
end


function i = indexof(x, v)
    i = length(v)/2 + 0.5;
    for j = 2:10
        if lerp(v, i) < x
            i = i + (length(v)-1)/(2^j);
        else
            i = i - (length(v)-1)/(2^j);
        end
    end
end

function y = lerp(x, i)
    w = i - floor(i);
    y = w * x(ceil(i)) + (1-w) * x(floor(i));
end

function plot_group(data)
    global CM;
    global LFS;
    
    figure;
    hold on;
    i = 1;
    for k = data.keys
        xy = data(k{1});
        if strcmp(k{1}, 'Spice')
            c = CM(3,:);
        else
            c = CM(i,:);
        end
        plot(xy(1,:), xy(2,:), 'LineWidth', 2, 'Color', c);
        i = i+1;
    end
    grid on;
    legend(data.keys, 'FontSize', LFS);
    tmp = get(gca, 'XTickLabel');  
    set(gca, 'XTickLabel', tmp, 'fontsize', LFS);
end

function xy = filter(json, select, where, group_by)
    xy = containers.Map;
    
    for i = 1:length(json)
        o = json(i);
        xy(num2str(o.(group_by))) = [];
    end
    
    for k = xy.keys
        where2 = where;
        where2{end+1} = group_by;
        where2{end+1} = k{1};
        
        xy(k{1}) = filter_unique(json, select, where2);
        
        if length(xy(k{1})) == 0
            remove(xy, k{1});
        end
    end
end

function xy = filter_unique(json, select, where)
    x = [];
    y = [];
    
    for i = 1:length(json)
        o = json(i);
        
        if ~isfield(o, select) | o.(select) == -1
            continue;
        end
            
        match = 1;
        for j = 1:2:length(where)
            if ~compare(o.(where{j}), where{j+1})
                match = 0;
                break;
            end
        end
        
        if match
            x = [x o.x_syn];
            y = [y o.(select)];
        end
    end
    
    if strcmp(select, 'simtime')
        y = y * 10;
    end
    xy = [x; y];
end

function eq = compare(a, b)
    if ischar(a) | ischar(b)
        eq = strcmp(lower(num2str(a)), lower(num2str(b)));
    else
        eq = a == b;
    end
end

function res = value_or(map, key, or)
    if ~isKey(map, key)
        map(key) = or;
    end
    
    res = map(key);
end



function colors = distinct_colors(n_colors,bg,func)
% DISTINGUISHABLE_COLORS: pick colors that are maximally perceptually distinct
%
% When plotting a set of lines, you may want to distinguish them by color.
% By default, Matlab chooses a small set of colors and cycles among them,
% and so if you have more than a few lines there will be confusion about
% which line is which. To fix this problem, one would want to be able to
% pick a much larger set of distinct colors, where the number of colors
% equals or exceeds the number of lines you want to plot. Because our
% ability to distinguish among colors has limits, one should choose these
% colors to be 'maximally perceptually distinguishable.'
%
% This function generates a set of colors which are distinguishable
% by reference to the 'Lab' color space, which more closely matches
% human color perception than RGB. Given an initial large list of possible
% colors, it iteratively chooses the entry in the list that is farthest (in
% Lab space) from all previously-chosen entries. While this 'greedy'
% algorithm does not yield a global maximum, it is simple and efficient.
% Moreover, the sequence of colors is consistent no matter how many you
% request, which facilitates the users' ability to learn the color order
% and avoids major changes in the appearance of plots when adding or
% removing lines.
%
% Syntax:
%   colors = distinguishable_colors(n_colors)
% Specify the number of colors you want as a scalar, n_colors. This will
% generate an n_colors-by-3 matrix, each row representing an RGB
% color triple. If you don't precisely know how many you will need in
% advance, there is no harm (other than execution time) in specifying
% slightly more than you think you will need.
%
%   colors = distinguishable_colors(n_colors,bg)
% This syntax allows you to specify the background color, to make sure that
% your colors are also distinguishable from the background. Default value
% is white. bg may be specified as an RGB triple or as one of the standard
% 'ColorSpec' strings. You can even specify multiple colors:
%     bg = {'w','k'}
% or
%     bg = [1 1 1; 0 0 0]
% will only produce colors that are distinguishable from both white and
% black.
%
%   colors = distinguishable_colors(n_colors,bg,rgb2labfunc)
% By default, distinguishable_colors uses the image processing toolbox's
% color conversion functions makecform and applycform. Alternatively, you
% can supply your own color conversion function.
%
% Example:
%   c = distinguishable_colors(25);
%   figure
%   image(reshape(c,[1 size(c)]))
%
% Example using the file exchange's 'colorspace':
%   func = @(x) colorspace('RGB->Lab',x);
%   c = distinguishable_colors(25,'w',func);
% Copyright 2010-2011 by Timothy E. Holy
  % Parse the inputs
  if (nargin < 2)
    bg = [1 1 1];  % default white background
  else
    if iscell(bg)
      % User specified a list of colors as a cell aray
      bgc = bg;
      for i = 1:length(bgc)
	bgc{i} = parsecolor(bgc{i});
      end
      bg = cat(1,bgc{:});
    else
      % User specified a numeric array of colors (n-by-3)
      bg = parsecolor(bg);
    end
  end
  
  % Generate a sizable number of RGB triples. This represents our space of
  % possible choices. By starting in RGB space, we ensure that all of the
  % colors can be generated by the monitor.
  n_grid = 30;  % number of grid divisions along each axis in RGB space
  x = linspace(0,1,n_grid);
  [R,G,B] = ndgrid(x,x,x);
  rgb = [R(:) G(:) B(:)];
  if (n_colors > size(rgb,1)/3)
    error('You can''t readily distinguish that many colors');
  end
  
  % Convert to Lab color space, which more closely represents human
  % perception
  if (nargin > 2)
    lab = func(rgb);
    bglab = func(bg);
  else
    C = makecform('srgb2lab');
    lab = applycform(rgb,C);
    bglab = applycform(bg,C);
  end
  % If the user specified multiple background colors, compute distances
  % from the candidate colors to the background colors
  mindist2 = inf(size(rgb,1),1);
  for i = 1:size(bglab,1)-1
    dX = bsxfun(@minus,lab,bglab(i,:)); % displacement all colors from bg
    dist2 = sum(dX.^2,2);  % square distance
    mindist2 = min(dist2,mindist2);  % dist2 to closest previously-chosen color
  end
  
  % Iteratively pick the color that maximizes the distance to the nearest
  % already-picked color
  colors = zeros(n_colors,3);
  lastlab = bglab(end,:);   % initialize by making the 'previous' color equal to background
  for i = 1:n_colors
    dX = bsxfun(@minus,lab,lastlab); % displacement of last from all colors on list
    dist2 = sum(dX.^2,2);  % square distance
    mindist2 = min(dist2,mindist2);  % dist2 to closest previously-chosen color
    [~,index] = max(mindist2);  % find the entry farthest from all previously-chosen colors
    colors(i,:) = rgb(index,:);  % save for output
    lastlab = lab(index,:);  % prepare for next iteration
  end
end
function c = parsecolor(s)
  if ischar(s)
    c = colorstr2rgb(s);
  elseif isnumeric(s) && size(s,2) == 3
    c = s;
  else
    error('MATLAB:InvalidColorSpec','Color specification cannot be parsed.');
  end
end
function c = colorstr2rgb(c)
  % Convert a color string to an RGB value.
  % This is cribbed from Matlab's whitebg function.
  % Why don't they make this a stand-alone function?
  rgbspec = [1 0 0;0 1 0;0 0 1;1 1 1;0 1 1;1 0 1;1 1 0;0 0 0];
  cspec = 'rgbwcmyk';
  k = find(cspec==c(1));
  if isempty(k)
    error('MATLAB:InvalidColorString','Unknown color string.');
  end
  if k~=3 || length(c)==1,
    c = rgbspec(k,:);
  elseif length(c)>2,
    if strcmpi(c(1:3),'bla')
      c = [0 0 0];
    elseif strcmpi(c(1:3),'blu')
      c = [0 0 1];
    else
      error('MATLAB:UnknownColorString', 'Unknown color string.');
    end
  end
end