%-------------------------------  Q1 --------------------------------------
%% Q1.1
clear; close all; clc;
vRest = -60;
v = -80:0.01:10;
u = vRest - v;
alpha_n = (.1 * u + 1)./(exp(1 + .1 * u) - 1) / 10;
beta_n = .125 * exp(u/80);
alpha_m = (u+25) ./ (exp(2.5+.1*u)-1)/10;
beta_m = 4*exp(u/18);
alpha_h = .07 * exp(u/20);
beta_h = 1 ./ (1+exp(3 + .1*u));
% time constants
tau_n = 1 ./ (alpha_n + beta_n);
tau_m = 1 ./ (alpha_m + beta_m);
tau_h = 1 ./ (alpha_h + beta_h);

figure;
hold on; grid minor;
plot(v, tau_h, 'LineWidth', 2)
plot(v, tau_m, 'LineWidth', 2)
plot(v, tau_n, 'LineWidth', 2)
title('time constants', 'Interpreter','latex')
legend('\tau_h', '\tau_m', '\tau_n')

% steady state values
n_inf = alpha_n ./ (alpha_n + beta_n);
m_inf = alpha_m ./ (alpha_m + beta_m);
h_inf = alpha_h ./ (alpha_h + beta_h);

figure;
hold on; grid minor;
plot(v, h_inf, 'LineWidth', 2)
plot(v, m_inf, 'LineWidth', 2)
plot(v, n_inf, 'LineWidth', 2)
title('steady state values', 'Interpreter','latex')
legend('$$h_{\infty}$$', '$$m_{\infty}$$', '$$n_{\infty}$$', 'Interpreter', 'latex')
%% Q2.1
dt = 0.01; % Simulation time step
Duration = 200; % Simulation length
T = ceil(Duration/dt);
t = (1:T) * dt; % Simulation time points in ms
I = zeros(1, T); % in uA, external stimulus (external current)
I(1:T) = 200; % an input current pulse
[v, ~, ~, ~] = simulate_HH(dt, I, T);
figure;
plot(t, v)
xlabel('Time(ms)', 'Interpreter','latex')
ylabel('v(t)', 'Interpreter','latex')
%% Q5.1
% !!!!!!!!!!!!!!!!! LONG RUNTIME !!!!!!!!!!!!!!!!!
dt = 0.01; % Simulation time step
Duration = 2000; % Simulation length
T = ceil(Duration/dt);
t = (1:T) * dt; %#ok<NASGU> % Simulation time points in ms
external_input = linspace(6, 112, 100);
amp = zeros(size(external_input)); % amplitude
FR  = zeros(size(external_input)); % firing rate

for count = 1:length(external_input)
    count %#ok<NOPTS> 
    I = external_input(count) * ones(1, T); % in uA, external stimulus (external current)
    [v, ~, ~, ~] = simulate_HH(dt, I, T);
    [pks,locs] = findpeaks(v);
    amp(count) = mean(pks(pks>=-20));
    FR(count) = sum(pks>-20) / (Duration*1e-3);
end

figure;
plot(external_input, amp)
xlabel('External Current($$\mu$$A)', 'Interpreter','latex')
ylabel('Amplitude(mv)', 'Interpreter','latex')
title('Amplitude vs. External Current', 'Interpreter','latex')
figure;
plot(external_input, FR)
xlabel('External Current($$\mu$$A)', 'Interpreter','latex')
ylabel('Firing Rate(Hz)', 'Interpreter','latex')
title('Firing Rate vs. External Current', 'Interpreter','latex')
%% Q7.1
dt = 0.01; % Simulation time step
Duration = 2000; % Simulation length
T = ceil(Duration/dt);
t = (1:T) * dt; % Simulation time points in ms
I = linspace(1, 200, T);
[v, ~, ~, ~] = simulate_HH(dt, I, T);
figure;
plot(t, v)
xlabel('Time(ms)', 'Interpreter','latex')
ylabel('v(t)', 'Interpreter','latex')
% figure;
% plot(t, I)
% xlabel('Time(ms)', 'Interpreter','latex')
% ylabel('I(t)', 'Interpreter','latex')
%% Q8.1
% very low input current
dt = 0.01; % Simulation time step
Duration = 2000; % Simulation length
T = ceil(Duration/dt);
t = (1:T) * dt; %#ok<NASGU> % Simulation time points in ms
I = 2*ones(1, T);
[v, n, m, h] = simulate_HH(dt, I, T);

figure;
subplot(1, 3, 1)
plot(v, n); grid minor;
xlabel('voltage(mv)', 'Interpreter','latex')
ylabel('n', 'Interpreter','latex')
subplot(1, 3, 2)
plot(v, m); grid minor;
xlabel('voltage(mv)', 'Interpreter','latex')
ylabel('m', 'Interpreter','latex')
subplot(1, 3, 3)
plot(v, h); grid minor;
xlabel('voltage(mv)', 'Interpreter','latex')
ylabel('h', 'Interpreter','latex')
sgtitle('$$I = 2\mu A$$', 'Interpreter', 'latex')

% normal input current
I = 8*ones(1, T);
[v, n, m, h] = simulate_HH(dt, I, T);

figure;
subplot(1, 3, 1)
plot(v, n); grid minor;
xlabel('voltage(mv)', 'Interpreter','latex')
ylabel('n', 'Interpreter','latex')
subplot(1, 3, 2)
plot(v, m); grid minor;
xlabel('voltage(mv)', 'Interpreter','latex')
ylabel('m', 'Interpreter','latex')
subplot(1, 3, 3)
plot(v, h); grid minor;
xlabel('voltage(mv)', 'Interpreter','latex')
ylabel('h', 'Interpreter','latex')
sgtitle('$$I = 8\mu A$$', 'Interpreter', 'latex')

% very high input current
I = 200*ones(1, T);
[v, n, m, h] = simulate_HH(dt, I, T);

figure;
subplot(1, 3, 1)
plot(v, n); grid minor;
xlabel('voltage(mv)', 'Interpreter','latex')
ylabel('n', 'Interpreter','latex')
subplot(1, 3, 2)
plot(v, m); grid minor;
xlabel('voltage(mv)', 'Interpreter','latex')
ylabel('m', 'Interpreter','latex')
subplot(1, 3, 3)
plot(v, h); grid minor;
xlabel('voltage(mv)', 'Interpreter','latex')
ylabel('h', 'Interpreter','latex')
sgtitle('$$I = 200\mu A$$', 'Interpreter', 'latex')
%% Q9.1
dt = 0.001; % Simulation time step
Duration = 2000; % Simulation length
T = ceil(Duration/dt);
t = (1:T) * dt; % Simulation time points in ms
% triangular input
f = 10;
I = 10*sawtooth(2*pi*f*t*1e-3,1/2);
[v, ~, ~, ~] = simulate_HH(dt, I, T);
figure;
subplot(2, 1, 1)
plot(t, v)
xlabel('Time(ms)', 'Interpreter','latex')
ylabel('v(t)', 'Interpreter','latex')
subplot(2, 1, 2)
plot(t, I)
xlabel('Time(ms)', 'Interpreter','latex')
ylabel('External Current($$\mu$$A)', 'Interpreter','latex')
%% sinusoidal input
f = 25;
I = 10*sin(2*pi*f*t*1e-3);
% I = 10*ones(1, T);
[v, ~, ~, ~] = simulate_HH(dt, I, T);
figure;
subplot(2, 1, 1)
plot(t, v)
xlabel('Time(ms)', 'Interpreter','latex')
ylabel('v(t)', 'Interpreter','latex')
subplot(2, 1, 2)
plot(t, I)
xlabel('Time(ms)', 'Interpreter','latex')
ylabel('External Current($$\mu$$A)', 'Interpreter','latex')
str = strcat('f = ', num2str(f), '(Hz)');
sgtitle(str, 'Interpreter', 'latex')
%% square input
f = 10;
I = 10*square(2*pi*f*t*1e-3,50);
[v, ~, ~, ~] = simulate_HH(dt, I, T);
figure;
subplot(2, 1, 1)
plot(t, v)
xlabel('Time(ms)', 'Interpreter','latex')
ylabel('v(t)', 'Interpreter','latex')
subplot(2, 1, 2)
plot(t, I)
xlabel('Time(ms)', 'Interpreter','latex')
ylabel('External Current($$\mu$$A)', 'Interpreter','latex')
%% chirp input
f0 = 10; f1 = 40; t1 = 1;
I = 10*chirp(t*1e-3, f0, t1, f1);
[v, ~, ~, ~] = simulate_HH(dt, I, T);
figure;
subplot(2, 1, 1)
plot(t, v)
xlabel('Time(ms)', 'Interpreter','latex')
ylabel('v(t)', 'Interpreter','latex')
subplot(2, 1, 2)
plot(t, I)
xlabel('Time(ms)', 'Interpreter','latex')
ylabel('External Current($$\mu$$A)', 'Interpreter','latex')
%% -------------------------------  Q2 --------------------------------------
%clc; clear all ; close all;
%% initiallization
dt = 0.01; % Simulation time step
Duration = 100; % Simulation length
T = ceil(Duration/dt);
t = (1:T) * dt; % Simulation time points in ms
%% Part 3-1 : Tonic Spiking
title1 = "Tonic Spiking";
x_label = " Time(ms)";
legend1 = "V(mv)";
lengend2="I(uA)";
a=0.02;        % due to the question
b=0.2;         % due to the question
c=-65;         % due to the question
d=2;           % due to the question
h=15;          % due to the question
I=h*[zeros(1,ceil(10/dt)),ones(1,T-ceil(10/dt))];
Izhikevich_simulator(a,b,c,d,dt,T,I,t,title1,x_label,legend1,lengend2,"on");
%% Part 3-2 : Phasic Spiking
title1 = "Phasic Spiking";
x_label =" Time(ms)";
legend1 = "V(mv)";
lengend2="I(uA)";
a=0.02;        % due to the question     
b=0.25;        % due to the question
c=-65;         % due to the question
d=6;          % due to the question
h=1;        % due to the question
I=h*[zeros(1,ceil(10/dt)),ones(1,T-ceil(10/dt))];
Izhikevich_simulator(a,b,c,d,dt,T,I,t,title1,x_label,legend1,lengend2,"on");
%% Part 3-3 : Tonic Bursting
title1 = "Tonic Bursting";
x_label =" Time(ms)";
legend1 = "V(mv)";
lengend2="I(uA)";
a=0.02;        % due to the question
b=0.2;        % due to the question
c=-50;         % due to the question
d=2;        % due to the question
h=15;        % due to the question
I=h*[zeros(1,ceil(10/dt)),ones(1,T-ceil(10/dt))];
Izhikevich_simulator(a,b,c,d,dt,T,I,t,title1,x_label,legend1,lengend2,"on");
%% Part 3-4 : Phasic Bursting
title1 = "Phasic Bursting";
x_label =" Time(ms)";
legend1 = "V(mv)";
lengend2="I(uA)";
a=0.02;        % due to the question
b=0.25;        % due to the question
c=-55;        % due to the question
d=0.05;        % due to the question
h=0.6;        % due to the question
I=h*[zeros(1,ceil(10/dt)),ones(1,T-ceil(10/dt))];
Izhikevich_simulator(a,b,c,d,dt,T,I,t,title1,x_label,legend1,lengend2,"on");
%% Part 3-4 : Mixed Model
title1 = "Mixed Model";
x_label =" Time(ms)";
legend1 = "V(mv)";
lengend2="I(uA)";
dt = 0.01; % Simulation time step
Duration = 200; % Simulation length
T = ceil(Duration/dt);
t = (1:T) * dt; % Simulation time points in ms
a=0.02;        % due to the question
b=0.2;        % due to the question
c=-55;        % due to the question
d=4;        % due to the question
h=10;        % due to the question
I=h*[zeros(1,ceil(10/dt)),ones(1,T-ceil(10/dt))];
Izhikevich_simulator(a,b,c,d,dt,T,I,t,title1,x_label,legend1,lengend2,"on");
%% Part 4 :v-u phasic spiking pattern
title1 = "phasic spiking";
x_label =" Time(ms)";
legend1 = "V(mv)";
lengend2="I(uA)";
dt = 0.01; % Simulation time step
Duration = 100; % Simulation length
T = ceil(Duration/dt);
t = (1:T) * dt; % Simulation time points in ms
a=0.02;        
b=0.25;        
c=-65;        
d=6;        
h=1;        
I=h*[zeros(1,ceil(10/dt)),ones(1,T-ceil(10/dt))];
[v,u]=Izhikevich_simulator(a,b,c,d,dt,T,I,t,title1,x_label,legend1,lengend2,"off");
syms V
dV=0.04*V^2+5*V+140+h; 
dU=b*V;
figure();
plot(v,u,"b");
grid on;
xlabel("v");
ylabel("u");
ylim([-17,-9]);
tt = strcat('u-v pahse plane for ',' Duration = ',num2str(Duration));
title("u-v pahse plane")
%% -------------------------------  Q3 --------------------------------------
%clc ; clear all ;close all;
%% Part 1
beta = 0.5;
gamma = 1;
syms x
F(x)=beta*(1+tanh(x*gamma));
figure()
fplot(F,[-5 5]);
title("CDF of distribution for \gamma = "+num2str(gamma))
xlabel("x")
grid on
f(x)=diff(F,x);
figure()
fplot(f,[-5 5]);
title("PDF of distribution for \gamma = "+num2str(gamma))
xlabel("x")
grid on
%% Part 2
dt = 0.01;
Duration = 20;
T = ceil(Duration/dt);
t = (1:T) * dt;
V_rest = -70;
V_th = -45;
V_max = 30;
tau_m = 20;
tau_spike = 2.5;
tau_ref = 5; 
R = 1;
legend1=[];
figure();
hold on
for j=[5 20 50 80 110]
    I = ones(1,T) * j; 
    v = ones(1,T) * V_rest; 
    i = 2;
    while i<=T
        p=double(F(v(i-1)-V_th));
        sp=binornd(1,double(p));
        if (sp>0)
            v(i:i+( tau_spike/dt)) = linspace(v(i-1), V_max , (tau_spike/dt+ 1));
            v(i+ tau_spike/dt + 1 :i +  (tau_spike + tau_ref)/dt) = V_rest;
            i = i + (tau_spike + tau_ref)/dt;
        else
            v(i) = v(i-1) + dt * (-v(i-1) + R*I(i-1))/tau_m;
        end
        i = i + 1;
    end
    plot(t,v(1:T));
    legend1=[legend1,"RI = "+num2str(j*R)];
end
grid on
title("membrain spike noisy model")
xlabel("T(ms)")
ylabel("V(mv)")
legend(legend1)
hold off
%% part 3-3
dt = 0.1;
Duration = 1000;
T = ceil(Duration/dt);
t = (1:T) * dt;
I = ones(1,T) * 10; 
v = ones(1,T) * V_rest; 
i = 2;
SP_voltage=[];
while i<=T
    p=double(F(v(i-1)-V_th));
    sp=binornd(1,double(p));
    if (sp>0)
        SP_voltage=[SP_voltage,v(i-1)];
        v(i:i+( tau_spike/dt)) = linspace(v(i-1), V_max , (tau_spike/dt+ 1));
        v(i+ tau_spike/dt + 1 :i +  (tau_spike + tau_ref)/dt) = V_rest;
        i = i + (tau_spike + tau_ref)/dt;
    else
        v(i) = v(i-1) + dt * (-v(i-1) + R*I(i-1))/tau_m;
    end
    i = i + 1;
end
figure;
histogram(SP_voltage,10);
title("firing Voltages Histogram for RI = "+num2str(R*I(1))+" & \gamma = "+num2str(gamma)+" & Duration = "+num2str(Duration) );
xlabel("V(mv)");
grid on;
%% PART 4
dt = 0.1;
Duration = 500;
T = ceil(Duration/dt);
t = (1:T) * dt;
R=1;
F=[];
I=4:300;
for j=I
    I_ext = ones(1,T) * j; 
    v = ones(1,T) * V_rest; 
    i = 2;
    count=0;
    while i<=T
        p=double(beta*(1+tanh((v(i-1)-V_th)*gamma)));
        sp=binornd(1,double(p));
        if (sp>0)
            v(i:i+( tau_spike/dt)) = linspace(v(i-1), V_max , (tau_spike/dt+ 1));
            v(i+ tau_spike/dt + 1 :i +  (tau_spike + tau_ref)/dt) = V_rest;
            i = i + (tau_spike + tau_ref)/dt;
            count=count+1;
        else
            v(i) = v(i-1) + dt * (-v(i-1) + R*I_ext(i-1))/tau_m;
        end
        i = i + 1;
    end
    F=[F,1000/Duration*count];
end
figure;
plot(I,F);
xlabel("I");
ylabel("firing rate");
title("F-I visualization");
grid on
%% -------------------------------  Q4 --------------------------------------
clc; clear all ; close all;
clc
clear
figure
% initial parameters
noip = 15;
interval = 60;
Mee = 1.25;
Mei = -1;
Mie = 1;
Mii = 0;
Ye = -10;
Yi = 10;
Te = 0.01;
Ti = 0.05;
f = @(t,Y) [(-Y(1)+ramp(Mee*Y(1)+Mei*Y(2)-Ye))/Te;(-Y(2)+ramp(Mie*Y(1)+Mii*Y(2)-Yi))/Ti];
y1 = linspace(-interval,interval,20);
y2 = linspace(-interval,interval,20);
% creates two matrices one for all the x-values on the grid, and one for
% all the y-values on the grid. Note that x and y are matrices of the same
% size and shape, in this case 20 rows and 20 columns
[x,y] = meshgrid(y1,y2);
u = zeros(size(x));
v = zeros(size(x));
% we can use a single loop over each element to compute the derivatives at
% each point (y1, y2)
t=0; % we want the derivatives at each point at t=0, i.e. the starting time
for i = 1:numel(x)
Yprime = f(t,[x(i); y(i)]);
u(i) = Yprime(1);
v(i) = Yprime(2);
end
quiver(x,y,u,v,'r');
xlabel('V_E')
ylabel('V_I')
% axis tight equal;
hold on
for i = 1:noip
[ts,ys] = ode45(f,[0,50],[rand()*interval*((-1)^floor(rand()*interval)); ...
rand()*interval*((-1)^floor(rand()*interval))]);
plot(ys(:,1),ys(:,2),'b')
plot(ys(1,1),ys(1,2),'bo') % starting point
plot(ys(end,1),ys(end,2),'ks') % ending point
xlim([-interval interval]);
ylim([-interval interval]);
end
syms t
fplot((t+Ye-Mee*t)/Mei,'m','LineWidth',2);
fplot((Yi-Mie*t)/(Mii-1),'g','LineWidth',2);
title("phase plane")
hold('off')
%% -------------------------------  Q5 --------------------------------------
clc; clear all ; close all ; 
%% part 2 (A , C) 
%%% initialization
stim_dur = 200e-3; 
Impulse_periode = 10e-3;
dt = 0.01e-3;
t = 0:dt:stim_dur;
EL = -70e-3;
tau_m = 20e-3;
tau_peak = 10e-3;
IR = 25e-3;
r_m = 100e-3;
V_tresh = -54e-3;
V_rest = -80e-3;
V_max = 0e-3 ;
%%% Excitatory
V_0 = -60e-3;
Ve = V_0 * ones(size(t)); 
p = zeros(size(t));
p(1:Impulse_periode/dt:end) = 1;
Es = 0e-3;
K = t/tau_peak.*exp(1-t/tau_peak);
g = conv(K,p);
i = 2;
while i <=size(t,2)
    DV = (-((Ve(i-1)-EL) + g(i-1) * (Ve(i-1)-Es) * r_m) + IR) / tau_m;
    Ve(i) = Ve(i-1) + DV*dt;
    if(Ve(i)>V_tresh)
        Ve(i) = V_max;
        Ve(i+1) = V_rest;
        i = i+1 ;
    end
    i = i+1;
end
%%% inhibitory
Es = -80e-3;
Vi = V_0 * ones(size(t));
i = 2;
while i <=size(t,2)
    DV = (-((Vi(i-1)-EL) + g(i-1) * (Vi(i-1)-Es) * r_m) + IR) / tau_m;
    Vi(i) = Vi(i-1) + DV*dt;
    if(Vi(i)>V_tresh)
        Vi(i) = V_max;
        Vi(i+1) = V_rest;
        i = i+1 ;
    end
    i = i+1;
end
figure();
subplot(2,1,1);
plot(1000*t,1000*Ve,'b');
title("Excitatory for V_0 = " + num2str(1000*V_0) + " & Impulse period = "+num2str(1000*Impulse_periode),'color','b');
xlabel("Time(ms)");
ylabel("Voltage(mV)");
grid on;
subplot(2,1,2)
plot(1000*t,1000*Vi,'k');
title("Inhibitory for V_0 = " + num2str(1000*V_0) + " & Impulse period = "+num2str(1000*Impulse_periode),'color','k');
xlabel("Time(ms)");
ylabel("Voltage(mV)");
grid on;
sgtitle("Membrane potential",'color','r')
%% PART 3 
%%% initiallization
stim_dur = 400e-3; 
dt = 0.1e-3;
t = 0:dt:stim_dur;
%%% Excitatory
A = 10;
V1 = -80e-3*ones(size(t));
V2 = -60e-3*ones(size(t));
Es = 0e-3;
K = t/tau_peak.*exp(1-t/tau_peak);
p1 = zeros(size(V2));
p1(V2==V_max) = A;
g1 = conv(K,p1);
p2 = zeros(size(V1));
p2(V1==V_max) = A;
g2 = conv(K,p2);
i = 2;
j = 2;
while i <=size(t,2) || j <=size(t,2)
    if(i <=size(t,2) && i<=j)
        DV1 = (-((V1(i-1)-EL) + g1(i-1) * (V1(i-1)-Es) * r_m) + IR) / tau_m;
        V1(i) = V1(i-1) + DV1*dt;
        if(V1(i) > V_tresh && i<size(t,2) -1 )
            V1(i) = V_max;
            V1(i+1) = V_rest;
            
            p2 = zeros(size(V1));
            p2(V1==V_max) = A;
            g2 = conv(K,p2);
            i = i+1;
        end
        i = i+1;
    elseif(j <=size(t,2))
        DV2 = (-((V2(j-1)-EL) + g2(i-1) * (V2(j-1)-Es) * r_m) + IR) / tau_m;
        V2(j) = V2(j-1) + DV2*dt;
        if(V2(j) > V_tresh && j<size(t,2) -1) 
            V2(j) = V_max;
            V2(j+1) = V_rest;
            p1 = zeros(size(V2));
            p1(V2==V_max) = A;
            g1 = conv(K,p1);
            j = j+1;
        end
        j = j+1;
    end
end
figure()
subplot(2,1,1)
plot(1000*t,1000*V1,'Color','r');
hold on
plot(1000*t,1000*V2,'Color','b');
title("Excitatory synapsis",'color','b');
xlabel("Time(ms)");
ylabel("V(mV)");
grid on;
legend("V_1","V_2");
%%% Inhibitory
V1 = -80e-3*ones(size(t));
V2 = -60e-3*ones(size(t));
Es = -80e-3;
p1 = zeros(size(V2));
p1(V2==V_max) = A;
g1 = conv(K,p1);
p2 = zeros(size(V1));
p2(V1==V_max) = A;
g2 = conv(K,p2);
i = 2;
j = 2;
while i <=size(t,2) || j <=size(t,2)
    if(i <=size(t,2) && i<=j)
        DV1 = (-((V1(i-1)-EL) + g1(i-1) * (V1(i-1)-Es) * r_m) + IR) / tau_m;
        V1(i) = V1(i-1) + DV1*dt;
        if(V1(i) > V_tresh)
            V1(i) = V_max;
            V1(i+1) = V_rest;
            
            p2 = zeros(size(V1));
            p2(V1==V_max) = A;
            g2 = conv(K,p2);
            i = i+1;
        end
        i = i+1;
    elseif(j <=size(t,2))
        DV2 = (-((V2(j-1)-EL) + g2(i-1) * (V2(j-1)-Es) * r_m) + IR) / tau_m;
        V2(j) = V2(j-1) + DV2*dt;
        if(V2(j) > V_tresh)
            V2(j) = V_max;
            V2(j+1) = V_rest;
            p1 = zeros(size(V2));
            p1(V2==V_max) = A;
            g1 = conv(K,p1);
            j = j+1;
        end
        j = j+1;
    end
end
subplot(2,1,2)
plot(1000*t,1000*V1,'Color','r');
hold on
plot(1000*t,1000*V2,'Color','b');
title("Inhibitory synapsis",'color','b');
xlabel("Time(ms)");
ylabel("V(mV)");
grid on;
legend("V_1","V_2");
sgtitle("Neuron potential",'color','r')
