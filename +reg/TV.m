classdef TV
    
    properties
        lam % lagrange multiplier for TV regularization
        type % TV norm type (L1 or iso)
    end
    
    methods
        function obj = TV(lam,type)
            % constructor function for TV regularizer

            % set labmda
            if nargin < 1 || isempty(lam)
                lam = 1; % default lam is 1
            end
            obj.lam = lam;

            % set norm type
            if nargin < 2 || isempty(type)
                type = 'l1';
            end
            obj.type = type;

        end
        
        function x_p = prox(obj,x,niter)
            % solve for x_p using a primal-dual approach

            % get size
            sz = size(x);
            nd = ndims(x);
            if (nd==2 && sz(2)==1)
                nd = 1;
            end

            % if lam is a scalar, apply it to all dimensions
            if length(obj.lam)==1
                obj.lam = obj.lam*ones(1,nd);
            elseif length(obj.lam)~=nd
                error('lam must be 1xnd or a scalar for all dimensions')
            end
            obj.lam(obj.lam==0) = eps; % protection from divide by 0

            % set default niter to 5
            if nargin < 3 || isempty(niter)
                niter = 50;
            end

            % initialize variables
            P = cell(nd,1); % finite difference images
            for d = 1:nd
                P{d} = L_adj(x,d);
            end
            R = P;
            t_k_1 = 1;

            % loop through iterations
            for i = 1:niter

                % store old values
                P_old = P;
                t_k = t_k_1;

                % compute the gradient
                D = x;
                for d = 1:nd % loop through dimensions
                    D = D - obj.lam(d)*L_fwd(R{d},d);
                end

                % take a step towards negative of the gradient
                for d = 1:nd
                    Qd = L_adj(D,d);
                    P{d} = R{d} + 1/(4*nd*obj.lam(d))*Qd;
                end

                % calculate projection
                switch obj.type
                    case 'iso'
                        P = project_iso(P,sz);
                    case 'l1'
                        P = project_l1(P);
                    otherwise
                        error('invalid type: %s',obj.type);
                end

                % calculate t_{k+1}
                t_k_1 = (1 + sqrt(1+4*t_k^2))/2;

                % update R
                for d = 1:nd
                    R{d} = P{d} + (t_k-1)/t_k_1 * (P{d} - P_old{d});
                end

            end

            % calculate x_p
            x_p = x;
            for i = 1:d
                x_p = x - obj.lam(d)*L_fwd(P{d},d);
            end

        end

        function n = norm(obj,x)
            % function to calculate TV norm
            switch obj.type
                case 'iso'
                    n = tvnorm_iso(x,obj.lam);
                case 'l1'
                    n = tvnorm_l1(x,obj.lam);
                otherwise
                    error('invalid type: %s',obj.type);
            end
        end
    end
end

%% define finite differencing operators L and L^T:
% reference section 4A in:
% Beck and Teboulle, “Fast Gradient-Based Algorithms for Constrained Total
% Variation Image Denoising and Deblurring Problems.”
%
% here, P is the set of finite difference matrices
% i.e. for 2D case: P = {p_ij, q_ij}, where
% p_ij = x_ij - x_i+1,j, i = 1, ..., m-1, j = 1, ..., n
% q_ij = x_ij - x_i,j+1, i = 1, ..., m, j = 1, ..., n-1
%
% matrix D is one of these "finite difference" matrices for a given
% dimension dim
% i.e. D_1 = L_adj(x,1) = x_ij - x_i+1,j; D_2 = L_fwd(x,2) = x_ij - x_i,j+1

function D = L_adj(x,dim)
    % zero-pad differencing dimension
    padsz = zeros(ndims(x),1);
    padsz(dim) = 1;

    % compute finite difference along dimension dim
    D = padarray(-diff(x,1,dim), padsz, 0, 'post');
end

function x = L_fwd(D, dim)
    % get size of x
    sz = size(D);

    % get indicies for addition and subtraction
    idx1 = repmat({':'}, 1, ndims(D));
    idx1{dim} = 1:sz(dim)-1;
    idx2 = idx1;
    idx2{dim} = idx1{dim}+1;

    % back-calculate x from finite difference operators
    x = zeros(sz);
    x(idx1{:}) = x(idx1{:}) + D(idx1{:});
    x(idx2{:}) = x(idx2{:}) - D(idx1{:});
end

%% define projection steps for iso and l1 TV
% reference remark 4.2 in:
% Beck and Teboulle, “Fast Gradient-Based Algorithms for Constrained Total
% Variation Image Denoising and Deblurring Problems.”

function P_proj = project_l1(P)
    % project onto set P1:
    % 2D case: P = {p,q} --> P1 = {r,s}, where
    % r_ij = p_ij / max{1,|p_ij|}
    % s_ij = q_ij / max{1,|q_ij|}

    P_proj = cell(size(P));

    % L1 projection
    for d = 1:length(P)
        P_proj{d} = P{d} ./ max(1,abs(P{d}));
    end

end

function P_proj = project_iso(P)
    % project onto set P1:
    % 2D case: P = {p,q} --> P1 = {r,s}, where
    % r_ij = p_ij / max{1,sqrt( p_ij^2 + q_ij^2 )}
    % s_ij = q_ij / max{1,sqrt( p_ij^2 + q_ij^2 )}

    P_proj = cell(size(P));

    % compute sum of squares of P
    P_sos = zeros(size(P{1}));
    for d = 1:length(P)
        P_sos = P_sos + P{d}.^2;
    end

    % isotropic projection
    for d = 1:length(P)
        P_proj{d} = P{d} ./ max(1,sqrt(P_sos));
    end

end

%% define norms for l1 and iso TV
% reference problem 2.2 in:
% Beck and Teboulle, “Fast Gradient-Based Algorithms for Constrained Total
% Variation Image Denoising and Deblurring Problems.”
%
% here, TV_I represents the isotropic TV norm, and TV_L1 the l1-based
% anisotropic TV norm
% i.e. for the 2D case:
% TV_I(x) = sum_all(sqrt(sum{P_d^2, for d = 1:2}))
% TV_L1(x) = sum_all(sum{abs(P_d), for d = 1:2})

function TV_I = tvnorm_iso(x,lam)

    % compute sum of squares of P
    P_sos = zeros(length(lam),1);
    for d = 1:length(lam)
        P_d = L_adj(x,d);
        P_sos = P_sos + lam(d)*P_d.^2;
    end

    % calculate norm
    TV_I = sum(vec(sqrt(P_sos)));

end

function TV_L1 = tvnorm_l1(x,lam)

    % compute sum of absolute values as norm
    TV_L1 = 0;
    for d = 1:length(lam)
        P_d = L_adj(x,d);
        TV_L1 = TV_L1 + lam(d)*sum(abs(P_d(:)));
    end
    
end