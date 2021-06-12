%%
close all;
clear all;
clc;

results = [
    jsondecode(fileread('spice.json'));
    jsondecode(fileread('bsim.json'));
    jsondecode(fileread('neurongpu.json'));
    jsondecode(fileread('genn.json'));
];

global CM;
CM = [
   0 178 191;
   149 232 12;
   255 194 13;
   255 0 0
] / 255;

global TFS; global LFS;
TFS = 16;
LFS = 14;

%% Sim. time as function of network size (single GPU)
plot_group(filter(results, 'simtime', {'x_gpus' 1 'model' 'vogels'}, 'sim'));
title('Vogels');
xlabel('Synapse Count', 'FontSize', LFS);
ylabel('Real Time\div Biological Time (x)', 'FontSize', LFS);
xticks([0:0.5:3] .* 1e9);
xticklabels({'0' '0.5B' '1B' '1.5B' '2B' '2.5B' '3B'});
%saveas(gcf, 'simtime_vogels.eps', 'epsc');

plot_group(filter(results, 'simtime', {'x_gpus' 1 'model' 'brunel'}, 'sim'));
title('Brunel');
xlabel('Synapse Count', 'FontSize', LFS);
ylabel('Real Time\div Biological Time (x)', 'FontSize', LFS);
xticks([0:1:3 5 10] .* 1e9);
xticklabels({'0' '1B' '2B' '3B' '5B' '10B'});
set(gca, 'XScale', 'log');
%saveas(gcf, 'simtime_brunel.eps', 'epsc');

c = CM;
CM = CM([2 4],:);
plot_group(filter(results, 'simtime', {'x_gpus' 1 'model' 'brunel+'}, 'sim'));
title('Brunel+');
xlabel('Synapse Count', 'FontSize', LFS);
ylabel('Real Time\div Biological Time (x)', 'FontSize', LFS);
xticks([0:2.5:20] .* 1e8);
xticklabels({'0' '0.25B' '0.5B' '0.75B' '1B' '1.25B' '1.5B' '1.75B' '2B'});
yticks([1 3 5 10 100]);
yticklabels({'1' '3' '5' '10' '100'});
set(gca, 'YScale', 'log');
%saveas(gcf, 'simtime_brunel+.eps', 'epsc');
CM = c;

%% Setup time as a function of network size (vogels)
plot_group(filter(results, 'setuptime', {'x_gpus' 1 'model' 'vogels'}, 'sim'));
genn = filter(results, 'setuptime', {'model' 'vogels' 'sim' 'GeNN'}, 'sim');
xy = genn('GeNN');
plot(xy(1,:), xy(2,:) + 15, 'LineWidth', 2, 'Color', CM(2,:), 'LineStyle', '--');
title('Setup');
xlabel('Synapse Count', 'FontSize', LFS);
ylabel('Setup Time (s)', 'FontSize', LFS);
xticks([0:0.5:3] .* 1e9);
xticklabels({'0' '0.5B' '1B' '1.5B' '2B' '2.5B' '3B'});
yticks([0.1 1 10 100 500]);
yticklabels({'0.1' '1' '10' '100' '500'});
legend({'BSim' 'GeNN' 'NeuronGPU' 'Spice' 'GeNN /w comp.'}, 'Location', 'NorthEast');
set(gca, 'YScale', 'log');
%saveas(gcf, 'setuptime.eps', 'epsc');


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
        plot(xy(1,:), xy(2,:), 'LineWidth', 2, 'Color', CM(i,:));
        i = i+1;
    end
    grid on;
    legend(data.keys, 'FontSize', LFS, 'Location', 'NorthWest');
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