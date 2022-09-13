function [v, n, m, h] = simulate_HH(dt, I, T)
% constants
Cm = 1; % Membrane capacitance in micro Farads
gNa = 120; % in Siemens, maximum conductivity of Na+ Channel
gK = 36; % in Siemens, maximum conductivity of K+ Channel
gl = 0.3; % in Siemens, conductivity of leak Channel
ENa = 55; % in mv, Na+ nernst potential
EK = -72; % in mv, K+ nernst potential
El = -49.4; % in mv, nernst potential for leak channel
vRest = -60; % in mv, resting potential

alpha_n = @(u) (.1 * u + 1)./(exp(1 + .1 * u) - 1) / 10;
alpha_m = @(u) (u+25) ./ (exp(2.5+.1*u)-1)/10;
alpha_h = @(u) .07 * exp(u/20);
beta_n = @(u) .125 * exp(u/80);
beta_m = @(u) 4*exp(u/18);
beta_h = @(u) 1 ./ (1+exp(3 + .1*u));
% initial values
v = zeros(1, T); % output voltage
v(1) = vRest;
n = zeros(1, T); % probability of K+  activation gate being open
n(1) = alpha_n(vRest - v(1)) / (alpha_n(vRest - v(1)) + beta_n(vRest - v(1)));
m = zeros(1, T); % probability of Na+ activation gate being open
m(1) = alpha_m(vRest - v(1)) / (alpha_m(vRest - v(1)) + beta_m(vRest - v(1)));
h = zeros(1, T); % probability of NA+ inactivation gate being open
h(1) = alpha_h(vRest - v(1)) / (alpha_h(vRest - v(1)) + beta_h(vRest - v(1)));
% 2nd order Runge-Kutta
k1 = zeros(4, 1);
k2 = zeros(4, 1);
for i = 1:T-1
    k1(1) = dt * (-1)/Cm * (gl*(v(i)-El) + gK*n(i)^4*(v(i)-EK) + gNa*m(i)^3*h(i)*(v(i)-ENa)-I(i));
    tau_n = 1 / (alpha_n(vRest - v(i)) + beta_n(vRest - v(i)));
    n_inf = alpha_n(vRest - v(i)) / (alpha_n(vRest - v(i)) + beta_n(vRest - v(i)));
    k1(2) = dt * 1/tau_n * (-n(i) + n_inf);
    tau_m = 1 / (alpha_m(vRest - v(i)) + beta_m(vRest - v(i)));
    m_inf = alpha_m(vRest - v(i)) / (alpha_m(vRest - v(i)) + beta_m(vRest - v(i)));
    k1(3) = dt * 1/tau_m * (-m(i) + m_inf);
    tau_h = 1 / (alpha_h(vRest - v(i)) + beta_h(vRest - v(i)));
    h_inf = alpha_h(vRest - v(i)) / (alpha_h(vRest - v(i)) + beta_h(vRest - v(i)));
    k1(4) = dt * 1/tau_h * (-h(i) + h_inf);

    k2(1) = dt * (-1)/Cm * (gl*((v(i)+k1(1))-El) + gK*(n(i)+k1(2))^4*((v(i)+k1(1))-EK) + gNa*(m(i)+k1(3))^3*(h(i)+k1(4))*((v(i)+k1(1))-ENa)-I(i));
    tau_n = 1 / (alpha_n(vRest - (v(i)+k1(1))) + beta_n(vRest - (v(i)+k1(1))));
    n_inf = alpha_n(vRest - (v(i)+k1(1))) / (alpha_n(vRest - (v(i)+k1(1))) + beta_n(vRest - (v(i)+k1(1))));
    k2(2) = dt * 1/tau_n * (-(n(i)+k1(2)) + n_inf);
    tau_m = 1 / (alpha_m(vRest - (v(i)+k1(1))) + beta_m(vRest - (v(i)+k1(1))));
    m_inf = alpha_m(vRest - (v(i)+k1(1))) / (alpha_m(vRest - (v(i)+k1(1))) + beta_m(vRest - (v(i)+k1(1))));
    k2(3) = dt * 1/tau_m * (-(m(i)+k1(3)) + m_inf);
    tau_h = 1 / (alpha_h(vRest - (v(i)+k1(1))) + beta_h(vRest - (v(i)+k1(1))));
    h_inf = alpha_h(vRest - (v(i)+k1(1))) / (alpha_h(vRest - (v(i)+k1(1))) + beta_h(vRest - (v(i)+k1(1))));
    k2(4) = dt * 1/tau_h * (-(h(i)+k1(4)) + h_inf);

    v(i+1) = v(i) + (k1(1)+k2(1))/2;
    n(i+1) = n(i) + (k1(2)+k2(2))/2;
    m(i+1) = m(i) + (k1(3)+k2(3))/2;
    h(i+1) = h(i) + (k1(4)+k2(4))/2;
end





















