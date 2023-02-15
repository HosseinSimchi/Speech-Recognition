%Hossein Simchi, 98443119
clear
clc
F0 = dlmread('C:\\Users\\Lenovo\\Desktop\\F0.txt');
F1 = dlmread('C:\\Users\\Lenovo\\Desktop\\F1.txt');
F2 = dlmread('C:\\Users\\Lenovo\\Desktop\\F2.txt');
F0_a = F0(1:6);
F1_a = F1(1:6);
F2_a = F2(1:6);

F0_e = F0(7:12);
F1_e = F1(7:12);
F2_e = F2(7:12);

F0_o = F0(13:18);
F1_o = F1(13:18);
F2_o = F2(13:18);

F0_A = F0(19:24);
F1_A = F1(19:24);
F2_A = F2(19:24);

F0_i = F0(25:30);
F1_i = F1(25:30);
F2_i = F2(25:30);

F0_u = F0(31:36);
F1_u = F1(31:36);
F2_u = F2(31:36);


title('Vowel Variability Single speaker')
xlabel('F1 Frequency (Hz)')
ylabel('F2 Frequency (Hz)')
hold on
scatter(F1_a,F2_a,'^');
scatter(F1_e,F2_e,'h');
scatter(F1_A,F2_A,'o');
scatter(F1_o,F2_o,'d');
scatter(F1_i,F2_i,'p');
scatter(F1_u,F2_u,'*');
hold off