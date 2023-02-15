% Hello, my name is Hossein Simhci, 98443119

m1 = 0;
m2 = 6;
w = [-8:0.001:8];
f1 = 2;
f2 = 1;
y1 = sin(w*f1*((m2+1)/2)) + sin(w*f2*((m2+1)/2)); 
y2 = sin((w/2)*f1) + sin((w/2)*f2); 
z = (1/(m2+1))*((y1)./(y2)).* exp(-1*1i*w*m2/2);
x1 = abs(z);
x2 = angle(z);
subplot(211)
plot(w,x1,'b')
set(gca,'XTick',-pi:pi/2:pi)
set(gca,'XTickLabel',{'-pi','-pi/2','0','pi/2','pi'})
grid; xlabel('w'); ylabel('Magnitude');
title("Hossein Simchi - 98443119")
subplot(212)
plot(w,x2,'r')
set(gca,'XTick',-pi:pi/2:pi)
set(gca,'XTickLabel',{'-pi','-pi/2','0','pi/2','pi'})
grid; xlabel('w'); ylabel('Phase');
title("Hossein Simchi - 98443119")