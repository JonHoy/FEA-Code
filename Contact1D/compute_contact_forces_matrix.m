function [contact_forces, contact_sides] = compute_contact_forces_matrix(XV, idt, num_domains, lambda)
  
  domain_spans = zeros(num_domains, 2);
  for idomain = 1:num_domains
    domain_spans(idomain, 1) = XV{idomain}(1, idt);
    domain_spans(idomain, 2) = XV{idomain}(end, idt);
  end  

  contact_forces = zeros(num_domains, num_domains);
  contact_sides = ones(num_domains, num_domains, 'int32') * -1;
  for idomain = 1:num_domains
    x1_left = domain_spans(idomain, 1);
    x1_right = domain_spans(idomain, 2);
    for jdomain = 1:num_domains
      x2_left = domain_spans(jdomain, 1);
      x2_right = domain_spans(jdomain, 2);
      if idomain ~= jdomain  
        [contact_force, contact_side] = ...
          compute_contact_forces_1d(x1_left, x1_right, x2_left, x2_right, lambda);

        contact_forces(idomain, jdomain) = contact_force;
        contact_sides(idomain, jdomain) = contact_side;
      end  
    end
  end
end

%[---1---] [---2---]
function [contact_force, contact_side] = compute_contact_forces_1d(...
  x1_left, x1_right, x2_left, x2_right, lambda)
  
  contact_force = 0.0;
  
  % if no contact between domain 1 and domain 2, contact_side = -1, 
  % if contact with domain 2 is on the left side of domain 1, contact_side = 0
  % if contact with domain 2 is on the right side of domain 1, contact_side = 1
  contact_side = -1;
  gap = 0.0;
  if x2_left < x1_right && x1_left < x2_left
    gap = x1_right - x2_left;
    contact_side = 1;
    disp('contact detected!')
  elseif x1_right > x2_right && x2_right > x1_left
    gap = x2_right > x1_left;
    contact_side = 0;
    disp('contact detected!')
  else
    gap = 0.0;
  end

  % use the penalty method
  contact_force = lambda * gap ^ 2;

end
