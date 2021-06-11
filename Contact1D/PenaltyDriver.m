
A = 1.0e-6;
E = 1e9;
rho = 1000.0;

a0 = 0.0;
a1 = 0.0;

gamma = 0.5;
beta = 0.25;

lambda = 1.0e4;

num_domains = 2;

nx = [2, 2];
v0 = [100.0, -100.0];
initial_span = cell(2,1);

initial_span{1} = [-1.0, -0.5];
initial_span{2} = [0.5, 1.0];


dt = 1.0e-7;
t_start = -0.2e-3;
t_end = 0.8e-3;

nt = (t_end - t_start) / dt + 1; 
t_end_actual = round(nt - 1) * dt + t_start;
tV = linspace(t_start, t_end_actual, nt);

TV = cell(num_domains, 1);
XV = cell(num_domains, 1);
VV = cell(num_domains, 1);
AV = cell(num_domains, 1);
F_extV = cell(num_domains, 1);
F_intV = cell(num_domains, 1);

for idomain = 1:num_domains
  TV{idomain} = tV;
  
  xv_initial = ...
    linspace(initial_span{idomain}(1), initial_span{idomain}(2), nx(idomain));
  XV{idomain} = zeros(nx(idomain), nt);
  VV{idomain} = zeros(nx(idomain), nt);
  AV{idomain} = zeros(nx(idomain), nt);
  
  F_intV{idomain} = zeros(nx(idomain), nt);
  F_extV{idomain} = zeros(nx(idomain), nt);

  % set initial conditions
  XV{idomain}(:, 1) = xv_initial;
  VV{idomain}(:, 1) = v0(idomain); 

end

for idt = 1:(nt-1)
  
  % compute contact forces
  [contact_forces, contact_sides] = compute_contact_forces_matrix(XV, idt, num_domains, lambda);

  for idomain = 1:num_domains
    f_int = F_intV{idomain}(:, idt);
    f_ext = F_extV{idomain}(:, idt);
    xV = XV{idomain}(:, idt);
    vV = VV{idomain}(:, idt);
    aV = AV{idomain}(:, idt);

    % Assemble the new mass, damping, and stiffness matrices
    [K, M, C] = Assembly(xV, A, E, rho, a0, a1);
    
    % Update the external load vector with the contact forces
    for jdomain = 1:num_domains
      f_contact = contact_forces(idomain, jdomain);
      contact_side = contact_sides(idomain, jdomain);
      if contact_side == 0
        f_ext(1) = f_contact;
      elseif contact_side == 1
        f_ext(end) = -f_contact;  
      end
    end
    
    % perform newmark integration
    f_net = f_int + f_ext;
    [aV_new, vV_new, xV_new] = newmark(M, C, K, f_net, xV, vV, aV, dt, gamma, beta);

    uV = xV_new - xV;    
    
    % deform mesh and update load vector
    [xV_new, f_int_new] =  Update_mesh_and_load_vector(f_int, xV, uV, K);
    F_intV{idomain}(:, idt + 1) = f_int_new; 
    XV{idomain}(:, idt + 1) = xV_new;
    VV{idomain}(:, idt + 1) = vV_new;
    AV{idomain}(:, idt + 1) = aV_new;
  end  
end
