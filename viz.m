f = fopen("~/brunel.txt");
n = fgetl(f); n = str2num(n);

width = ceil(sqrt(n));
i = zeros(width * width, 1);

l = fgetl(f);
while ischar(l)
    spikes = split(l, ',');
    spikes = uint32(arrayfun(@(x) str2num(x{1}), spikes)) + 1;
    
    i(:) = 0;
    i(spikes) = 1;
    
    imshow(reshape(i, width, width)', 'InitialMagnification', 400);
    
    l = fgetl(f);
end