clear all;
close all;
results = jsondecode(fileread("spice.json"));

gsim = "Spice";%lif_unit_test";

% Sim. time as function of network size (single GPU)
[x, y] = filter(results, "simtime", {"sim" gsim "x_gpus" 1 "model" "vogels"});
figure;
plot_simtime("Vogels", x, y);

[x, y] = filter(results, "simtime", {"sim" gsim "x_gpus" 1 "model" "brunel"});
figure;
plot_simtime("Brunel", x, y);

[x, y] = filter(results, "simtime", {"sim" gsim "x_gpus" 1 "model" "brunel+"});
figure;
plot_simtime("Brunel+", x, y);
xlabel("#Synapses (M)");

% Sim. time as a function of network size (single GPU, sparse)
[x, y] = filter(results, "simtime", {"sim" gsim "x_gpus" 1 "model" "synth_0.00156_0.005_1"});
figure;
plot_simtime("Synth", x, y);
%xlabel("#Synapses (B) (#Neurons (M))");
%xticklabels({"0 (0)", "0.5 (0.57)", "1 (0.8)", "1.5 (0.98)", "2 (1.13)", "2.5 (1.27)", "3 (1.4)"});

% Constr. time as a function of network size
figure;
hold on;
[x, y] = filter(results, "setuptime", {"sim" gsim "x_gpus" 1 "model" "synth_0.05_0.001_1"});
plot_simtime("Setup Time", x, y);
[x, y] = filter(results, "setuptime", {"sim" gsim "x_gpus" 2 "model" "synth_0.05_0.001_1"});
plot_simtime("Setup Time", x, y);
[x, y] = filter(results, "setuptime", {"sim" gsim "x_gpus" 4 "model" "synth_0.05_0.001_1"});
plot_simtime("Setup Time", x, y);
[x, y] = filter(results, "setuptime", {"sim" gsim "x_gpus" 8 "model" "synth_0.05_0.001_1"});
plot_simtime("Setup Time", x, y);
ylabel("Setup Time (s)");
legend("Ours (1 GPU)", "Ours (2 GPUs)", "Ours (4 GPUs)", "Ours (8 GPUs)");

% Sim. time as a function of network size (multi-GPU)
figure;
hold on;
[x, y] = filter(results, "simtime", {"sim" gsim "x_gpus" 1 "model" "vogels"});
plot_simtime("Vogels", x, y);
[x, y] = filter(results, "simtime", {"sim" gsim "x_gpus" 2 "model" "vogels"});
plot_simtime("Vogels", x, y);
[x, y] = filter(results, "simtime", {"sim" gsim "x_gpus" 4 "model" "vogels"});
plot_simtime("Vogels", x, y);
[x, y] = filter(results, "simtime", {"sim" gsim "x_gpus" 8 "model" "vogels"});
plot_simtime("Vogels", x, y);
legend("1 GPU", "2 GPUs", "4 GPUs", "8 GPUs", "Location", "East");

figure;
hold on;
[x, y] = filter(results, "simtime", {"sim" gsim "x_gpus" 1 "model" "brunel"});
plot_simtime("Brunel", x, y);
[x, y] = filter(results, "simtime", {"sim" gsim "x_gpus" 2 "model" "brunel"});
plot_simtime("Brunel", x, y);
[x, y] = filter(results, "simtime", {"sim" gsim "x_gpus" 4 "model" "brunel"});
plot_simtime("Brunel", x, y);
[x, y] = filter(results, "simtime", {"sim" gsim "x_gpus" 8 "model" "brunel"});
plot_simtime("Brunel", x, y);
legend("1 GPU", "2 GPUs", "4 GPUs", "8 GPUs", "Location", "East");

figure;
hold on;
[x, y] = filter(results, "simtime", {"sim" gsim "x_gpus" 1 "model" "brunel+"});
plot_simtime("Brunel+", x, y);
[x, y] = filter(results, "simtime", {"sim" gsim "x_gpus" 2 "model" "brunel+"});
plot_simtime("Brunel+", x, y);
[x, y] = filter(results, "simtime", {"sim" gsim "x_gpus" 4 "model" "brunel+"});
plot_simtime("Brunel+", x, y);
[x, y] = filter(results, "simtime", {"sim" gsim "x_gpus" 8 "model" "brunel+"});
plot_simtime("Brunel+", x, y);
legend("1 GPU", "2 GPUs", "4 GPUs", "8 GPUs", "Location", "East");
xlabel("#Synapses (M)");

% Speedup
speedup = [1 1];
scaleup = [1 1];
i = [8 6 5.5];
ii = [9 6 5.5];
for igpu = 1:3
    s1 = [];
    s2 = [];
    
    for model = {"vogels", "brunel"}%, "brunel+"}
        [a, x] = filter(results, "simtime", {"sim" gsim "x_gpus" 1 "model" model{1}});
        [b, y] = filter(results, "simtime", {"sim" gsim "x_gpus" 2^igpu "model" model{1}});
        s1 = [s1 x(end) / lerp(y, indexof(a(end), b))];
        s2 = [s2 x(end) / lerp(y, indexof(x(end), y)) * lerp(b, indexof(x(end), y)) / a(end)];
    end
    
    speedup = [speedup; s1];
    scaleup = [scaleup; s2];
end
figure;
bar(speedup);
title("Speedup");
xlabel("#GPUs");
xticklabels({"1", "2", "4", "8"});
ylabel("Speedup");
legend("Vogels", "Brunel", "Brunel+", "Location", "Northwest");
set(gca, "YGrid", "on");

% Sizeup
figure;
bar(scaleup);
title("Scaleup");
xlabel("#GPUs");
xticklabels({"1", "2", "4", "8"});
ylabel("Scaleup");
legend("Vogels", "Brunel", "Brunel+", "Location", "Northwest");
set(gca, "YGrid", "on");


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

function plot_simtime(name, x, y)
    plot(x, y, "LineWidth", 2);
    title(name);
    xlabel("#Synapses (B)");
    ylabel("Simulation Time (s)");
    grid on;
end

function [nsyn, t] = filter(json, select, where)
    nsyn = [];
    t = [];
    
    for j = 1:length(json)
        o = json(j);
        
        if ~isfield(o, select) | o.(select) == -1
            continue;
        end
            
        match = 1;
        for i = 1:2:length(where)
            if o.(where{i}) ~= where{i+1}
                match = 0;
                break;
            end
        end
        
        if match
            nsyn = [nsyn; o.x_syn];
            t = [t; o.(select)];
        end
    end
end