classdef LLR
    
    properties
        lam
        patch_size
        prox_avg
    end
    
    methods
        function obj = LLR(lam,patch_size,prox_avg)
            
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

            % set default proximal average flag
            if nargin < 3 || isempty(prox_avg)
                prox_avg = 1;
            end
            obj.prox_avg = prox_avg;

        end
        
        function x_p = prox(obj,x)

            % compute the Casorati matrix
            C = Patch_fwd(x,obj.patch_size,obj.prox_avg);
            C_lr = zeros(size(C));

            % perform SVD on the Casorati matrix of each patch
            for s = 1:size(C,4)
                for p = 1:size(C,3)
                    [U, S, V] = svd(C(:,:,p,s), 'econ');

                    % soft-threshold the singular values
                    S_lr = diag(max(diag(S) - obj.lam, 0));

                    % reconstruct the low-rank Casorati approximation
                    C_lr(:,:,p,s) = U * S_lr * V';
                end
            end

            % reconstruct the LLR approximation of x
            x_p = Patch_adj(C_lr,size(x),obj.prox_avg);

        end

        function n = norm(obj,x)

            % compute the Casorati matrix
            C = Patch_fwd(x,obj.patch_size,obj.prox_avg);
            
            % take the nuclear norm of C by summing its singular values
            n = 0;
            for s = 1:size(C,4)
                for p = 1:size(C,3)
                    [~,S,~] = svd(C(:,:,p), 'econ');
                    n = n + obj.lam*sum(diag(S));
                end
            end

        end

    end
end

%% define the Casorati patch operators
% for a given image timeseries x, the Casorati matrix C is build from
% patches of x. Each patch has a given Casorati matrix of size Ne x Nt,
% where Ne is the number of elements in each patch (i.e. patch_size^nd) and
% Nt is the number of time points of x

function C = Patch_fwd(x, patch_size, prox_avg)
    
    % get size
    sz = size(x);
    nd = ndims(x);
    
    % calculate sizes
    nt = sz(end); % assuming time is in last dimension
    np = floor(sz(1:nd-1)/patch_size);  % number of patches along each dim & stride
    if prox_avg
        ns = patch_size;
    else
        ns = 1;
    end

    % intitialize Casorati matrix
    C = zeros(patch_size^(nd-1), nt, prod(np), ns^nd);

    % extract patches and fill Casorati matrix
    for si = 1:ns
        for sj = 1:ns
            for pi = 1:np(1)
                for pj = 1:np(2)
                    % collection of indicies for current patch
                    I = mod(((pi-1)*patch_size+1:pi*patch_size) + si - 1, sz(1)) + 1;
                    J = mod(((pj-1)*patch_size+1:pj*patch_size) + sj - 1, sz(2)) + 1;

                    % extract current patch
                    if nd == 3 % 2D case
                        patch_idx = sub2ind(np,pi,pj);
                        shift_idx = sub2ind(ns*ones(1,nd),si,sj);
                        patch = x(I,J,:);
                        C(:,:,patch_idx, shift_idx) = reshape(patch,patch_size^2,nt);
                    elseif nd == 4 % 3D case
                        for sk = 1:ns
                            for pk = 1:np(3)
                                K = mod(((pk-1)*patch_size+1:pk*patch_size) + sk - 1, sz(3)) + 1;
                                patch_idx = sub2ind(np,pi,pj,pk);
                                shift_idx = sub2ind(ns*ones(1,nd),si,sj,sk);
                                patch = x(I,J,K,:);
                                C(:,:,patch_idx,shift_idx) = reshape(patch,patch_size^3,nt);
                            end
                        end
                    else
                        error('patch extraction only defined for 2D and 3D timeseries')
                    end

                end
            end
        end
    end

end

function x = Patch_adj(C, sz, prox_avg)
    
    % get sizes
    nd = length(sz);
    patch_size = sqrt(size(C,1));
    nt = size(C,2);
    np = floor(sz(1:nd-1)/patch_size);  % number of patches along each dim, assuming stride = 1
    if prox_avg
        ns = patch_size;
    else
        ns = 1;
    end

    % intitialize reconstructed image
    x = zeros(sz);

    % extract patches and fill Casorati matrix
    for si = 1:ns
        for sj = 1:ns
            for pi = 1:np(1)
                for pj = 1:np(2)
                    % collection of indicies for current patch
                    I = mod(((pi-1)*patch_size+1:pi*patch_size) + si - 1, sz(1)) + 1;
                    J = mod(((pj-1)*patch_size+1:pj*patch_size) + sj - 1, sz(2)) + 1;

                    % extract current patch
                    if nd == 3 % 2D case
                        patch_idx = sub2ind(np,pi,pj);
                        shift_idx = sub2ind(ns*ones(1,nd),si,sj);
                        patch = reshape(C(:,:,patch_idx,shift_idx),patch_size,patch_size,nt);
                        x(I,J,:) = x(I,J,:) + patch;
                    elseif nd == 4 % 3D case
                        for pk = 1:np(3)
                            K = mod(((pk-1)*patch_size+1:pk*patch_size) + sk - 1, sz(3)) + 1;
                            patch_idx = sub2ind(np,pi,pj,pk);
                            shift_idx = sub2ind(ns*ones(1,nd),si,sj,sk);
                            patch = reshape(C(:,:,patch_idx,shift_idx),patch_size,patch_size,patch_size,nt);
                            x(I,J,K,:) = x(I,J,K,:) + patch;
                        end
                    else
                        error('patch extraction only defined for 2D and 3D timeseries')
                    end

                end
            end
        end
    end

end