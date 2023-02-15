
clear
clc
%Hossein Simchi, 98443119

%Fs * 2   or   Fs / 2
[y,Fs] = audioread('C:\Users\Lenovo\Desktop\File.wav');
t = 1/Fs :1/Fs:size(y,1)/Fs;
sound(y,Fs);
sound(y,Fs/2);
audiowrite('C:\Users\Lenovo\Desktop\File_sampling_rate_nim.wav',y,Fs/2);
sound(y,2*Fs);
audiowrite('C:\Users\Lenovo\Desktop\File_sampling_rate_2.wav',y,Fs*2);


%Real part of FFT
x0 = fft(y);
r =  real(x0);
x1 = real(ifft(r));
audiowrite('C:\Users\Lenovo\Desktop\File_FFT_real_part.wav',x1,Fs);
plot(t,x1)
xlabel('Tims')
ylabel('Signal') 
title('Real part audio (Hossein Simchi, 98443119)')



%zero phase 
x0 = fft(y);
a = abs(x0);
theta = angle(x0);
q = a.*exp(1i*0);
x1 = real(ifft(q));
n = x1/(max(abs(x1)));
audiowrite('C:\Users\Lenovo\Desktop\File_phase_zero.wav',n,Fs)
plot(t,x1)
xlabel('Tims')
ylabel('Signal') 
title('Phase zero audio (Hossein Simchi, 98443119)')


%Twice fft
s = fft(fft(y));
r = real((s));
normal = r/(max(abs(r)));
audiowrite('C:\Users\Lenovo\Desktop\File_Twice_FFT.wav',normal,Fs);
plot(t,normal)
xlabel('Tims')
ylabel('Signal') 
title('Twice FFT (Hossein Simchi, 98443119)')


%Add Sin noise to signal
sin = sin(2*pi*500*t);
sin = reshape(sin,size(y));
noise = y + sin;
normal = noise/(max(abs(noise)));
audiowrite('C:\Users\Lenovo\Desktop\File_Sin_noise.wav',normal, Fs);
plot(t,normal)
xlabel('Tims')
ylabel('Signal') 
title('Sin noise (Hossein Simchi, 98443119)')



%Noise removal
sin = sin(2*pi*500*t);
sin = reshape(sin,size(y));
noise = y + sin;
a = fft(noise);
b = round(500*size(y,1)/Fs);
for i = b-600 : b + 600
    a(i) = 0;
end
b = round((Fs-500)*size(y,1)/Fs);

for i = b-600 : b + 600
    a(i) = 0;
end
audiowrite('C:\Users\Lenovo\Desktop\File_noise_removal.wav',real(ifft(a)),Fs);




