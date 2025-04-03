function result = detectTuringSpace(par, range1, range2, eps)
%detectTuringSpace checks the conditions of Turing instability for given parameters of the Lengyel-Epstein model.
%INPUT
%par: a structure containing model parameters (L, d, a, b, sigma) in fields.
%range1: (optional) eigenvalue range for the first dimension of the spatial variable. Default is 100.
%range2: (optional) eigenvalue range for the second dimension of the spatial variable. Default is 100.
%eps: (Optional) tolerance value for determining positivity. Default is 0.
%OUTPUT
%result: A boolean indicating whether Turing conditions are satisfied (true) or not (false).


L = par.L;  
d = par.d;
a = par.a;
b = par.b;
sigma = par.sigma;

if nargin < 2 || isempty(range1)
    range1 = 100;
end
if nargin < 3 || isempty(range2)
    range2 = 100;
end
if nargin < 4
    eps = 0;
end

dfdv = @(v,w) -((4 * w) / (v * v + 1) ...
    - (8 * v * v * w)/((v * v + 1) * (v * v + 1)) + 1) / sigma;
dfdw = @(v,w) -(4 * v) / (sigma * (v * v + 1));

dgdv = @(v,w) b * ((2 * v * v * w) / ((v * v + 1) * (v * v + 1)) ...
    - w / (v * v + 1) + 1);
dgdw = @(v,w) -(b * v) / (v * v + 1);

v0 = a / 5;
w0 = 1 + v0.^2;

J0 = [ dfdv(v0, w0) dfdw(v0, w0);
       dgdv(v0, w0) dgdw(v0, w0) ...
       ];

result = false;

for i=0:range1
    for j=0:range2
        A = J0 - diag([1 / sigma d]) * ((pi*i/L)^2 + (pi*j/L)^2);
        if i==0 && j==0
            if any(real(eig(A)) >= 0)
                return;
            end
        else
            if any(real(eig(A)) >= eps)
                result = true;
                return;
            end
        end
    end
end

end

