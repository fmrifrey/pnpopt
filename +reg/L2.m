classdef L2
    
    properties
        lam % lagrange multiplier for L2 regularization
    end
    
    methods
        function obj = L2(lam)
            % constructor function for L2 regularizer
            if nargin < 1 || isempty(lam)
                lam = 1; % default lam is 1
            end
            obj.lam = lam;
        end
        
        function x_p = prox(obj,x)
            % computes the proximal operation for L2 regularization
            % i.e. x_p = x / (1 + lambda)
            x_p = x / (1 + obj.lam);
        end

        function n = norm(obj,x)
            % computes the L2 norm of x
            % i.e. ||x||_2
            n = obj.lam*norm(x(:),2);
        end

    end
end

