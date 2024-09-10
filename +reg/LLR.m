classdef LLR
    
    properties
        lam
        patch_size
    end
    
    methods
        function obj = LLR(lam,patch_size)
            
            % set labmda
            if nargin < 1 || isempty(lam)
                lam = 1; % default lam is 1
            end
            obj.lam = lam;

            % set default patch size
            if nargin < 2 || isempty(patch_size)
                patch_size = 6;
            end
            obj.patch_size = patch_size;

        end
        
        function x_p = prox(obj,x)

            % compute the Casorati matrix
            C = Patch_fwd(x,obj.patch_size);
            C_lr = zeros(size(C));

            % perform SVD on the Casorati matrix of each patch
            for p = 1:size(C,3)
                [U, S, V] = svd(C(:,:,p), 'econ');

                % soft-threshold the singular values
                S_lr = diag(max(diag(S) - obj.lam, 0));

                % reconstruct the low-rank Casorati approximation
                C_lr(:,:,p) = U * S_lr * V';
            end

            % reconstruct the LLR approximation of x
            x_p = Patch_adj(C_lr,size(x));

        end

        function n = norm(obj,x)

            % compute the Casorati matrix
            C = Patch_fwd(x,obj.patch_size);
            
            % take the nuclear norm of C by summing its singular values
            n = 0;
            for p = 1:size(C,3)
                [~,S,~] = svd(C(:,:,p), 'econ');
                n = n + obj.lam*sum(diag(S));
            end

        end

    end
end

%% define the Casorati patch operators
% for a given image timeseries x, the Casorati matrix C is build from
% patches of x. Each patch has a given Casorati matrix of size Ne x Nt,
% where Ne is the number of elements in each patch (i.e. patch_size^nd) and
% Nt is the number of time points of x

function C = Patch_fwd(x, patch_size)
    
    % get size
    sz = size(x);
    nd = ndims(x);
    
    % calculate sizes
    nt = sz(end); % assuming time is in last dimension
    np = floor(sz(1:nd-1)/patch_size);  % number of patches along each dim, assuming stride = 1

    % intitialize Casorati matrix
    C = zeros(patch_size^(nd-1), nt, prod(np));
    
    % extract patches and fill Casorati matrix
    for i = 1:np(1)
        for j = 1:np(2)
            % collection of indicies for current patch
            I = (i-1)*patch_size+1:i*patch_size;
            J = (j-1)*patch_size+1:j*patch_size;

            % extract current patch
            if nd == 3 % 2D case
                patch_idx = sub2ind(np,i,j);
                patch = x(I,J,:);
                C(:,:,patch_idx) = reshape(patch,patch_size^2,nt);
            elseif nd == 4 % 3D case
                for k = 1:np(3)
                    K = (k-1)*patch_size+1:k*patch_size;
                    patch_idx = sub2ind(np,i,j,k);
                    patch = x(I,J,K,:);
                    C(:,:,patch_idx) = reshape(patch,patch_size^3,nt);
                end
            else
                error('patch extraction only defined for 2D and 3D timeseries')
            end

        end
    end

end

function x = Patch_adj(C, sz)
    
    % get sizes
    nd = length(sz);
    patch_size = sqrt(size(C,1));
    nt = size(C,2);
    np = floor(sz(1:nd-1)/patch_size);  % number of patches along each dim, assuming stride = 1

    % intitialize reconstructed image
    x = zeros(sz);
    
    % extract patches and fill Casorati matrix
    for i = 1:np(1)
        for j = 1:np(2)
            % collection of indicies for current patch
            I = (i-1)*patch_size+1:i*patch_size;
            J = (j-1)*patch_size+1:j*patch_size;

            % extract current patch
            if nd == 3 % 2D case
                patch_idx = sub2ind(np,i,j);
                patch = reshape(C(:,:,patch_idx),patch_size,patch_size,nt);
                x(I,J,:) = patch;
            elseif nd == 4 % 3D case
                for k = 1:np(3)
                    K = (k-1)*patch_size+1:k*patch_size;
                    patch_idx = sub2ind(np,i,j,k);
                    patch = reshape(C(:,:,patch_idx),patch_size,patch_size,patch_size,nt);
                    x(I,J,K,:) = patch;
                end
            else
                error('patch extraction only defined for 2D and 3D timeseries')
            end

        end
    end

end