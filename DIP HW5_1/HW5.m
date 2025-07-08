clc;
clear all;
close all;
image=im2gray(imread('Fig5.25.jpg'));
image=imresize(image,[256 256])

x=fspecial('gaussian',260,2);
y=(imfilter(image,x,'circular'));

F = zeros(size(image));
G = fftshift(fft2(y));

H = fftshift(fft2(x));

R=55;

for i=1:size(image,2)
    for j=1:size(image,1)
        di = i - size(image,2)/2;
        dj = j - size(image,1)/2;
        if di^2 + dj^2 <= R^2;
        F(j,i) = G(j,i)./H(j,i);
        end
    end
end

inverse_filtered = abs(ifftshift(ifft2(F)));
figure,imshow(inverse_filtered, []);
