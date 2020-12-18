clc;
close all;

%f = fopen("~/brunel.txt");
f = fopen("/media/dkb/data/google_drive/projects/BSim/release/GSim.log");
n = fgetl(f); n = str2num(n);

width = ceil(sqrt(n));
i = zeros(width * width, 1);

l = fgetl(f);
while ischar(l)
    i(:) = 0;
    
    if length(l) > 0
        spikes = split(l, ',');
        spikes = uint32(str2double(spikes)) + 1;
        i(spikes) = 1;
    end
    
    imshow(reshape(i, width, width)', 'InitialMagnification', 400);
    
    l = fgetl(f);
end