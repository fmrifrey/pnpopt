classdef TV
    
    properties
        lam % lagrange multiplier for TV regularization
        type % TV norm type (L1 or iso)
        dim % dimensions of TV regularization
    end
    
    methods
        function obj = TV(lam,type,dim)
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

            % set TV dimensions
            if nargin < 3
                dim = [];
            end
            obj.dim = dim;

        end
        
        function x_p = prox(obj,x,niter)
            % solve for x_p using a primal-dual approach

            % get size
            sz = size(x);
            nd = ndims(x);
            if (nd==2 && sz(2)==1)
                nd = 1;
            end

            % set default niter to 5
            if nargin < 3 || isempty(niter)
                niter = 50;
            end

            % initialize variables
            P = L_adj(x,obj.dim);
            R = P;
            t_k_1 = 1;

            % loop through iterations
            for i = 1:niter

                % store old values
                P_old = P;
                t_k = t_k_1;

                % compute gradient of objective fun
                D = x - obj.lam*L_fwd(R,obj.dim);
                Q = L_adj(D,obj.dim);

                % take a step towards negative of the gradient
                for d = 1:nd
                    P{d} = R{d} + 1/(4*nd*obj.lam)*Q{d};
                end

                % calculate projection
                switch obj.type
                    case 'iso'
                        P = prox_iso(P,sz,nd);
                    case 'l1'
                        P = prox_l1(P,nd);
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

            % calculate Y
            x_p = x - obj.lam*L_fwd(P,obj.dim);

        end

        function n = norm(obj,x)
            % function to calculate TV norm
            switch obj.type
                case 'iso'
                    n = tvnorm_iso(x,obj.dim);
                case 'l1'
                    n = tvnorm_l1(x,obj.dim);
                otherwise
                    error('invalid type: %s',obj.type);
            end
        end
    end
end

%% define finite differencing operators L and L':

function x = L_fwd(P,dim)
% P = cell array of projection matrices for each dimension
% dim = dimensions to apply finite differencing
    
    % set defaults dims to all
    nd = length(P);
    if nargin < 2 || isempty(dim)
        dim = 1:nd;
    end

    % get size
    sz = size(P{1});
    if dim(1) == 1
        sz(1) = sz(1)+1; % first dimension will be differenced - add 1
    end

    % initialize image
    x = zeros(sz);
    idcs1 = cell(nd,1);
    idcs2 = cell(nd,1);

    % loop through dimensions
    for d1 = dim
        % get indicies for addition and subtraction
        for d2 = 1:nd
            if d1 == d2
                idcs1{d2} = 1:sz(d2)-1;
                idcs2{d2} = 2:sz(d2);
            else
                idcs1{d2} = 1:sz(d2);
                idcs2{d2} = 1:sz(d2);
            end
        end

        % calculate
        % L(p,q)_i,j = p_i,j + q_i,j - p_i-1,j - q_i-1,j
        x(idcs1{:}) = x(idcs1{:}) + P{d1};
        x(idcs2{:}) = x(idcs2{:}) - P{d1};
    end

end

function P = L_adj(x, dim)
% x = image of any dimensions
% dim = dimensions to apply finite differencing

    % get size
    sz = size(x);
    nd = ndims(x);

    % set defaults dims to all
    if nargin < 2 || isempty(dim)
        dim = 1:nd;
    end

    % initialize cell array of finite diff matrices
    P = cell(nd,1);
    
    % loop through dimensions
    for d = 1:nd
        if ismember(d,dim)
            % calculate L'(x) = {p,q}
            % p_i,j = x_i,j - x_i+1,j
            % q_i,j = x_i,j - x_i,j+1
            P{d} = -diff(x,1,d);
        else
            P{d} = zeros(sz);
        end
    end

end

%% define proximal operators for iso and l1 TV

function P = prox_iso(P,sz,nd)

    % loop through dimensions
    A = zeros(sz); % root sum of squares
    for d = 1:nd
        padsz = zeros(nd,1);
        padsz(d) = size(A,d) - size(P{d},d);

        % add square of P to A
        A = A + padarray(P{d},padsz,0,'post').^2;
    end

    % take root of sum of squares
    A = sqrt(max(A,1));

    % loop through dimensions
    idcs = cell(nd,1);
    for d1 = 1:length(P)

        % get indicies for addition and subtraction
        for d2 = 1:nd
            if d1 == d2
                idcs{d2} = 1:sz(d2)-1;
            else
                idcs{d2} = 1:sz(d2);
            end
        end

        % divide P by rsos
        P{d1} = P{d1}./A(idcs{:});
    end

end

function P = prox_l1(P,nd)
    
    % loop through dimensions
    for d = 1:nd
        % divide P by absolute max
        P{d} = P{d} ./ (sign(P{d}) .* max(abs(P{d}), 1));
    end

end

%% define norms for l1 and iso TV

function tv = tvnorm_iso(x,dim)

    % get size
    sz = size(x);
    nd = ndims(x);
    if (nd==2 && sz(2)==1)
        nd = 1;
    end

    % calculate projection operators
    P = L_adj(x,dim);

    % loop through dimensions
    D = zeros(sz); % sum of squares matrix
    for d = 1:nd
        padsz = zeros(nd,1);
        padsz(d) = size(D,d) - size(P{d},d);

        % add its square to D
        D = D + padarray(P{d},padsz,0,'post').^2;
    end

    % calculate norm
    tv = sum(sqrt(D),'all');

end

function tv = tvnorm_l1(x,dim)

    % get size
    sz = size(x);
    nd = ndims(x);
    if (nd==2 && sz(2)==1)
        nd = 1;
    end

    % calculate projection operators
    P = L_adj(x,dim);

    % loop through dimensions
    tv = 0;
    for d = 1:nd
        % add l1 norm
        tv = tv + sum(abs(P{d}),'all');
    end
    
end