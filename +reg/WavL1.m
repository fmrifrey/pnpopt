classdef WavL1

    properties
        lam % lagrange multiplier for L1 regularization
        n_lvl % number of levels for wavelet dec
    end
    
    methods
        function obj = WavL1(lam,n_lvl)
            % constructor function for Wavelet L1 regularizer
            if nargin < 1 || isempty(lam)
                lam = 1; % default lambda is 1
            end
            obj.lam = lam;
        
            if nargin < 2 || isempty(n_lvl)
                n_lvl = 3;
            end
            obj.n_lvl = n_lvl;
        end
        
        function x_p = prox(obj, x)
            % compute the DWT
            [C,L] = wavedec(x(:),obj.n_lvl,'db4');

            % soft thresholding operation (L1 proximal operator)
            l1_prox = @(x) sign(x) .* max(abs(x) - obj.lam, 0);

            % apply l1 thresholding
            C = l1_prox(C);
        
            % inverse DWT to reconstruct x
            x_p = reshape(waverec(C,L,'db4'),size(x));

        end

        function n = norm(obj, x)
            % compute the DWT
            C = wavedec(x(:),obj.n_lvl,'db4');

            % computes the L1 norm of x
            % i.e. ||Wx||_1
            n = obj.lam*(sum(abs(C(:))));
        end

    end
end

