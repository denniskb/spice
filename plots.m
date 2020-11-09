close all;
results = jsondecode(fileread("results.json"));

% Sim. time as function of network size (single GPU)
[x, y] = filter(results, "simtime", {"sim" "samples" "x_gpus" 1 "model" "vogels"});
figure;
plot_simtime("Vogels", x, y);

[x, y] = filter(results, "simtime", {"sim" "samples" "x_gpus" 1 "model" "brunel"});
figure;
plot_simtime("Brunel", x, y);

[x, y] = filter(results, "simtime", {"sim" "samples" "x_gpus" 1 "model" "brunel+"});
figure;
plot_simtime("Brunel+", x, y);
xlabel("#Synapses (M)");

% Sim. time as a function of network size (single GPU, sparse)
[x, y] = filter(results, "simtime", {"sim" "samples" "x_gpus" 1 "model" "synth_0.00156_0.001_1"});
figure;
plot_simtime("Synth", x, y);
xlabel("#Synapses (B) (#Neurons (M))");
xticklabels({"0 (0)", "0.5 (0.57)", "1 (0.8)", "1.5 (0.98)", "2 (1.13)", "2.5 (1.27)", "3 (1.4)"});

% Constr. time as a function of network size
figure;
hold on;
[x, y] = filter(results, "setuptime", {"sim" "samples" "x_gpus" 1 "model" "synth_0.05_0.001_1"});
plot_simtime("Setup Time", x, y);
[x, y] = filter(results, "setuptime", {"sim" "samples" "x_gpus" 2 "model" "synth_0.05_0.001_1"});
plot_simtime("Setup Time", x, y);
[x, y] = filter(results, "setuptime", {"sim" "samples" "x_gpus" 4 "model" "synth_0.05_0.001_1"});
plot_simtime("Setup Time", x, y);
ylabel("Setup Time (s)");
legend("Ours (1 GPU)", "Ours (2 GPUs)", "Ours (4 GPUs)");

% Sim. time as a function of network size (multi-GPU)
figure;
hold on;
[x, y] = filter(results, "simtime", {"sim" "samples" "x_gpus" 1 "model" "vogels"});
plot_simtime("Vogels", x, y);
[x, y] = filter(results, "simtime", {"sim" "samples" "x_gpus" 2 "model" "vogels"});
plot_simtime("Vogels", x, y);
[x, y] = filter(results, "simtime", {"sim" "samples" "x_gpus" 4 "model" "vogels"});
plot_simtime("Vogels", x, y);
legend("1 GPU", "2 GPUs", "4 GPUs");

figure;
hold on;
[x, y] = filter(results, "simtime", {"sim" "samples" "x_gpus" 1 "model" "brunel"});
plot_simtime("Brunel", x, y);
[x, y] = filter(results, "simtime", {"sim" "samples" "x_gpus" 2 "model" "brunel"});
plot_simtime("Brunel", x, y);
[x, y] = filter(results, "simtime", {"sim" "samples" "x_gpus" 4 "model" "brunel"});
plot_simtime("Brunel", x, y);
legend("1 GPU", "2 GPUs", "4 GPUs");

figure;
hold on;
[x, y] = filter(results, "simtime", {"sim" "samples" "x_gpus" 1 "model" "brunel+"});
plot_simtime("Brunel+", x, y);
[x, y] = filter(results, "simtime", {"sim" "samples" "x_gpus" 2 "model" "brunel+"});
plot_simtime("Brunel+", x, y);
[x, y] = filter(results, "simtime", {"sim" "samples" "x_gpus" 4 "model" "brunel+"});
plot_simtime("Brunel+", x, y);
legend("1 GPU", "2 GPUs", "4 GPUs");
xlabel("#Synapses (M)");

% Speedup
speed = [1 1 1];
size = [1 1 1];
i = [6 5 5];
ii = [8 5 5];
for igpu = 1:2
    s = [];
    ss = [];
    
    [~, x] = filter(results, "simtime", {"sim" "samples" "x_gpus" 1 "model" "vogels"});
    [~, y] = filter(results, "simtime", {"sim" "samples" "x_gpus" 2^igpu "model" "vogels"});
    s = [s x(8)/y(i(igpu))];
    ss = [ss x(12)/y(12)];
    
    [~, x] = filter(results, "simtime", {"sim" "samples" "x_gpus" 1 "model" "brunel"});
    [~, y] = filter(results, "simtime", {"sim" "samples" "x_gpus" 2^igpu "model" "brunel"});
    s = [s x(8)/y(i(igpu))];
    ss = [ss x(12)/y(12)];
    
    [a, x] = filter(results, "simtime", {"sim" "samples" "x_gpus" 1 "model" "brunel+"});
    [b, y] = filter(results, "simtime", {"sim" "samples" "x_gpus" 2^igpu "model" "brunel+"});
    s = [s x(8)/y(ii(igpu))];
    ss = [ss x(12)/y(12)];
    
    speed= [speed; s];
    size = [size ; 2^igpu*ss];
end
figure;
bar(speed);
title("Speedup");
xlabel("#GPUs");
xticklabels({"1", "2", "4"});
ylabel("Speedup");
legend("Vogels", "Brunel", "Brunel+", "Location", "Northwest");
set(gca, "YGrid", "on");

% Sizeup
figure;
bar(size);
title("Sizeup");
xlabel("#GPUs");
xticklabels({"1", "2", "4"});
ylabel("Speedup");
legend("Vogels", "Brunel", "Brunel+", "Location", "Northwest");
set(gca, "YGrid", "on");



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
        o = json{j};
        
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