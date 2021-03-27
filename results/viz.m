clc;
close all;

%f = fopen("~/brunel.txt");
f = fopen("/media/dkb/data/google_drive/projects/ijcnn2020/build/release/test.txt");
n = fgetl(f); n = str2num(n);

width = ceil(sqrt(n));
i = zeros(width * width, 1);

l = fgetl(f);
total = 0;
while ischar(l)
    i(:) = 0;
    
    if length(l) > 0
        spikes = split(l, ',');
        spikes = str2double(spikes) + 1;
        i(spikes) = 1;
        total = total + length(spikes);
    end
    
    imshow(reshape(i, width, width)', 'InitialMagnification', 400);
    
    l = fgetl(f);
end

total/n/1000