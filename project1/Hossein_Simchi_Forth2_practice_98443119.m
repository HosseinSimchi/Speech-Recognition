% Hossein Simchi, 98443119

m1 = 3;
m2 = 3;
w = [-8:0.001:8];
z = 0;
for n = -m1:m2
z = z + exp(-1*1i*w*n);
end
z = (1/(m1+m2+1)) * z;
x1 = abs(z);
x2 = angle(z);
subplot(211)
plot(w,x1,'r')
set(gca,'XTick',-pi:pi/2:pi)
set(gca,'XTickLabel',{'-pi','-pi/2','0','pi/2','pi'})
grid; xlabel('w'); ylabel('Magnitude');
subplot(212)
plot(w,x2,'b')
set(gca,'XTick',-pi:pi/2:pi)
set(gca,'XTickLabel',{'-pi','-pi/2','0','pi/2','pi'})
grid; xlabel('w'); ylabel('Phase');