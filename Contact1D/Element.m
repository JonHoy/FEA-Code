function [Ke, Me, Ce] = Element(Le, A, E, rho, a0, a1)
% simple 1d bar element formulation
Ke = A * E / Le * [1, -1; -1, 1];  
% uses the consistent mass matrix formulation
Me = rho * A * Le / 6.0 * [2, 1; 1, 2];

Ce = a0 * Me + a1 * Ke;

end