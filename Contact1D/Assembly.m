function [K,M,C] = Assembly(xV, A, E, rho, a0, a1)
  Nx = length(xV) - 1;
  num_nodes = Nx + 1;
  K = zeros(num_nodes, num_nodes);
  M = zeros(num_nodes, num_nodes);
  C = zeros(num_nodes, num_nodes);

  for ielement = 1:Nx
    Le = abs(xV(ielement + 1) - xV(ielement));
    [Ke, Me, Ce] = Element(Le, A, E, rho, a0, a1);
    K(ielement:ielement+1, ielement:ielement+1) = ...
      K(ielement:ielement+1, ielement:ielement+1) + Ke;
    M(ielement:ielement+1, ielement:ielement+1) = ...
      M(ielement:ielement+1, ielement:ielement+1) + Me;
    C(ielement:ielement+1, ielement:ielement+1) = ...
      C(ielement:ielement+1, ielement:ielement+1) + Ce;      

  end  

  % make them sparse
  K = sparse(K);
  M = sparse(M);
  C = sparse(C);  

end

