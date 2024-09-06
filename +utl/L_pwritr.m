function L = L_pwritr(A,niter,tol)
% Estimates the Lipschitz constant of a given system A using the power
% iteration method

    % set defaults
    if nargin < 2 || isempty(niter)
        niter = 20;
    end

    if nargin < 3 || isempty(tol)
        tol = 0;
    end

    if isfield(A,'idim')
        sz = A.idim;
    else
        sz = size(A, 2);
    end

    % initialize variables
    x = randn(sz); % generate random vector
    x = x / norm(x(:)); % normalize initial vector
    L = 0; % initialize Lipschitz constant estimate

    for n = 1:niter

        % apply operator A
        y = A * x;
        x_new = A' * y;

        % compute Lipschitz estimate
        L_new = norm(x_new(:)) / norm(x(:));

        % check for convergence
        if abs(L_new - L) < tol
            break;
        end

        % update variables
        L = L_new;
        x = x_new / norm(x_new(:));

    end

end
