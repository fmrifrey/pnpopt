classdef L1
    
    properties
        lam % lagrange multiplier for L1 regularization
    end
    
    methods
        function obj = L1(lam)
            % xonstructor function for L1 regularizer
            if nargin < 1 || isempty(lam)
                lam = 1; % default lambda is 1
            end
            obj.lam = lam;
        end
        
        function x_p = prox(obj, x)
            % computes the proximal operation for L1 regularization
            % i.e. x_p = sign(x) * max(|x| - lambda, 0)
            x_p = sign(x) .* max(abs(x) - obj.lam, 0);
        end

        function n = norm(obj, x)
            % computes the L1 norm of x
            % i.e. ||x||_1
            n = obj.lam*sum(abs(x(:)));
        end
    end
end


