%https://www.sciencedirect.com/topics/engineering/newmark-method
%https://web.stanford.edu/group/frg/course_work/AA242B/CA-AA242B-Ch7.pdf

function [a_new, v_new, x_new] = newmark(M, C, K, f, x, v, a, dt, gamma, beta)
  A = M + gamma * dt * C + beta * dt ^ 2 * K;
  b1 = f;
  b2 = -C * (v + (1.0 - gamma) * dt * a); 
  b3 = -K * (x + dt * v + (0.5 - beta) * dt ^ 2 * a);
  b = b1 + b2 + b3;
  % solve the linear system
  % first check to see if all zeros in b
  max_b = max(abs(b));
  a_new = zeros(length(b), 1);
  if max_b > eps
   a_new = A \ b;
  end  
  
  v_new = v + (1 - gamma) * dt * a + gamma * dt * a_new;
  x_new = x + dt * v + 0.5 * dt ^ 2 * ((1 - 2 * beta) * a + 2 * beta * a_new);
  
end  
