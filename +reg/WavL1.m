classdef WavL1

    properties
        lam % lagrange multiplier for L1 regularization
    end
    
    methods
        function obj = WavL1(lam)
            % constructor function for Wavelet L1 regularizer
            if nargin < 1 || isempty(lam)
                lam = 1; % default lambda is 1
            end
            obj.lam = lam;
        end
        
        function x_p = prox(obj, x)
            % compute the DWT
            [ca,cd] = dwt(x(:),'haar');

            % soft thresholding operation (L1 proximal operator)
            l1_prox = @(x) sign(x) .* max(abs(x) - obj.lam, 0);

            % apply l1 thresholding
            ca = l1_prox(ca);
            cd = l1_prox(cd);
        
            % inverse DWT to reconstruct x
            x_p = reshape(idwt(ca,cd,'haar'),size(x));

        end

        function n = norm(obj, x)
            % compute the DWT
            [ca,cd] = dwt(x(:),'haar');

            % computes the L1 norm of x
            % i.e. ||Wx||_1
            n = obj.lam*(sum(abs(ca)+abs(cd)));
        end

    end
end

