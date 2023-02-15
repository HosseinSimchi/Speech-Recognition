% Hello my name is Hossein Simchi, 98443119

Fs = 16000;
F0 = 122.8;
F1 = 431;
F2 = 1624.8;


bw = 0.05;

T = 20*(1/15);
fs = 16000;
t = 0:1/fs:T-1/fs;

x = sawtooth(2*pi*250*t);

[a0,b0] = iirpeak(F0/8000,bw);
[a,b] = iirpeak(F1/8000,bw);
[c,d] = iirpeak(F2/8000,bw);

x1 = filter(a0,b0,x);
x2 = filter(a,b,x1);
x3 = filter(c,d,x2);

sound(x3,Fs);
audiowrite('C:\Users\Lenovo\Desktop\e_Simchi_98443119.wav',x3,Fs);


