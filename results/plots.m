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
plot_group(filter(results, 'simtime', {'model' 'vogels' 'x_gpus' 1}, 'sim'));
title('Vogels');
xlabel('Synapse Count', 'FontSize', LFS);
ylabel('Real Time\div Biological Time (x)', 'FontSize', LFS);
xticks([0:0.5:3] .* 1e9);
xticklabels({'0' '0.5B' '1B' '1.5B' '2B' '2.5B' '3B'});
saveas(gcf, 'simtime_vogels.eps', 'epsc');

%%
plot_group(filter(results, 'simtime', {'model' 'brunel' 'x_gpus' 1}, 'sim'));
title('Brunel');
xlabel('Synapse Count', 'FontSize', LFS);
ylabel('Real Time\div Biological Time (x)', 'FontSize', LFS);
ylim([0 6]);
xticks([0:1:3 5 10] .* 1e9);
xticklabels({'0' '1B' '2B' '3B' '5B' '10B'});
yticks([0.5 1 2 6]);
yticklabels({'0.5' '1' '2' '6'});
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
saveas(gcf, 'simtime_brunel.eps', 'epsc');

%%
c = CM;
CM = CM([2 4],:);
plot_group(filter(results, 'simtime', {'model' 'brunel+'}, 'sim'));
title('Brunel+');
xlabel('Synapse Count', 'FontSize', LFS);
ylabel('Real Time\div Biological Time (x)', 'FontSize', LFS);
xticks([0 0.5 1 2] .* 1e9);
xticklabels({'0' '0.5B' '1B' '2B'});
ylim([0 20]);
yticks([0:1:5 10 20]);
yticklabels({'0' '1' '2' '3' '4' '5' '10' '20'});
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
saveas(gcf, 'simtime_brunel+.eps', 'epsc');
CM = c;

%% Setup time as a function of network size (vogels)
plot_group(filter(results, 'setuptime', {'model' 'vogels' 'x_gpus' 1}, 'sim'));
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
l = legend({'BSim' 'GeNN' 'NeuronGPU' 'Spice' 'GeNN /w comp.'}, 'Location', 'NorthEast');
set(l, 'color', 'w');
set(gca, 'YScale', 'log');
saveas(gcf, 'setuptime.eps', 'epsc');

%% Speedups of various opt.
eager = filter(jsondecode(fileread('spice_eager.json')), 'simtime', {}, 'sim');
eager = eager('SpiceEager');
lazy  = filter(jsondecode(fileread('spice_lazy.json')), 'simtime', {}, 'sim');
lazy  = lazy('SpiceLazy');
event = filter(results, 'simtime', {'sim' 'Spice' 'model' 'brunel+'}, 'sim');
event = event('Spice');

figure;
bar([1 eager(2,end) / lazy(2,end) eager(2,end) / event(2,end)], 'FaceColor', CM(2,:));
tmp = gca;
tmp.YGrid = 'on';
title('Plastic Models', 'FontSize', TFS);
ylim([0 30]);
xlabel('Optimization', 'FontSize', LFS);
ylabel('Speedup (x)', 'FontSize', LFS);
xticklabels({'Eager' 'Lazy' 'Lazy+Event'});
legend({'Brunel+'}, 'Location', 'NorthWest', 'FontSize', LFS);
tmp = get(gca, 'XTickLabel');  
set(gca, 'XTickLabel', tmp, 'fontsize', LFS);
tmp = get(gca, 'YTickLabel');
set(gca, 'YTickLabel', tmp, 'fontsize', LFS);

saveas(gcf, 'speedup_plastic.eps', 'epsc');


naive = filter(jsondecode(fileread('spice_naive.json')), 'simtime', {}, 'model');
naive_vogels = naive('vogels');
naive_brunel = naive('brunel');
vogels = filter(results, 'simtime', {'sim' 'Spice' 'model' 'vogels'}, 'sim');
vogels = vogels('Spice');
brunel = filter(results, 'simtime', {'sim' 'Spice' 'model' 'brunel'}, 'sim');
brunel = brunel('Spice');

figure;
b = bar([1 1; naive_vogels(2,end) / vogels(2,end) naive_brunel(2,end) / brunel(2,end)]);
b(1).FaceColor = CM(1,:);
b(2).FaceColor = CM(3,:);
tmp = gca;
tmp.YGrid = 'on';
title('Static Models', 'FontSize', TFS);
xlabel('Optimization', 'FontSize', LFS);
ylabel('Speedup (x)', 'FontSize', LFS);
xticklabels({'Global' 'Shared'});
legend({'Vogels' 'Brunel'}, 'Location', 'NorthWest', 'FontSize', LFS);
tmp = get(gca, 'XTickLabel');  
set(gca, 'XTickLabel', tmp, 'fontsize', LFS);
tmp = get(gca, 'YTickLabel');
set(gca, 'YTickLabel', tmp, 'fontsize', LFS);

saveas(gcf, 'speedup_static.eps', 'epsc');


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
    l = legend(data.keys, 'FontSize', LFS, 'Location', 'NorthWest');
    set(l, 'color', 'none');
    tmp = get(gca, 'XTickLabel');  
    set(gca, 'XTickLabel', tmp, 'fontsize', LFS);
    tmp = get(gca, 'YTickLabel');
    set(gca, 'YTickLabel', tmp, 'fontsize', LFS);
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