function [f_new, xV_new] = Update_mesh_and_load_vector(f, xV, uV, K)
  % apply mesh deformation
  xV_new = xV + uV;
  % update the load vector
  f_new = f - K * uV;
end