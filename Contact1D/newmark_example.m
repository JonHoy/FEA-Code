clear all

M = [10, 0, 0; 0, 20, 0; 0, 0, 30];

K = [45, -20, -15; -20, 45, -25; -15, -25, 40];

C = 3e-2*K;

t_start = 0.0;
t_end = 3.0;
dt = 0.005;
omega = pi / 0.29; 

gamma = 0.5;
beta = 0.25;

n = (t_end - t_start) / dt + 1;

t = linspace(t_start, t_end, n);


F = zeros(3, n);
F(1, :) = 0;
F(2, :) = 0;
F(3, :) = 50 * sin(omega * t);
for i = 1:n
 if t(i) > pi/omega
   F(3,i) = 0.0;
 end
end


A = zeros(3, n);
V = zeros(3, n);
X = zeros(3, n);

for i = 1:n-1
  a = A(:, i);
  v = V(:, i);
  x = X(:, i);
  f = F(:, i);
 
  [a_new, v_new, x_new] = newmark(M, C, K, f, x, v, a, dt, gamma, beta); 
 
  A(:, i+1) = a_new;
  V(:, i+1) = v_new;
  X(:, i+1) = x_new;
 
end

for i = 1:3
  plot(t, X(i,:))
  hold on
end



