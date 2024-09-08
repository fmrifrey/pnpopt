function [x_star, cost] = nlCG(x0, A, b, varargin)

    % set defaults
    defaults.niter = 5; % number of iterations
    defaults.R = []; % regularizers
    defaults.betaMethod = 'Dai-Yuan'; % method for updating beta
    defaults.maxlsiter = 6; % maximum line search iterations
    defaults.alpha = 0.01; % step size factor for line search
    defaults.beta = 0.8; % step size factor for line search
    defaults.t0 = 0.0001; % initial step size
    defaults.update_fun = []; % iteration update fun

    % parse inputs
    args = vararg_pair(defaults,varargin);

    % initialize cost
    cost = zeros(1, args.niter+1);

    % define the cost function
    function c = cost_fun(x)
        % calculate data fidelity term
        c = 0.5 * norm(A * x - b, 2)^2;

        % add regularization norms
        if ~isempty(args.R) && iscell(args.R)
            for i = 1:length(R)
                c = c + args.R{i}.norm(x);
            end
        elseif ~isempty(args.R)
            c = c + args.R.norm(x);
        end
    end

    % initialize variables
    x_star = x0;
    g0 = reshape(A' * (A*x_star - b), size(x0)); % initial gradient
    f0 = cost_fun(x_star); % initial objective function value
    dx = -g0; % initial search direction = negative gradient
    cost(1) = f0;

    for n = 1:args.niter
        time_itr = tic; % start timer

        % line search to find optimal step size
        t = args.t0;
        f1 = cost_fun(x_star + t * dx);
        lsiter = 0;
        while (f1 > f0 - args.alpha * t * abs(g0(:)' * dx(:))) % Wolfe condition
            lsiter = lsiter + 1;
            t = t * args.beta;
            f1 = cost_fun(x_star + t*dx);
            if lsiter > args.maxlsiter
                break
            end
        end

        % update the solution x*
        x_star = x_star + t * dx;

        % apply the regularization
        if ~isempty(args.R) && iscell(args.R) % for multiple regularizers
            for j = 1:length(args.R)
                x_star = args.R{j}.prox(x_star);
            end
        elseif ~isempty(args.R) % for a single regularizer
            x_star = args.R.prox(x_star);
        end

        % calculate the new cost
        cost(n+1) = cost_fun(x_star);
        f0 = cost(n+1);

        % calculate the new gradient
        g1 = reshape(A' * (A * x_star - b), size(x0));

        % calculate b_k
        switch lower(args.betaMethod)
            case 'dai-yuan'
                b_k = (g1(:)'*g1(:)) / (dx(:)' * (g1(:)-g0(:)));
            case 'fletcher-reeves'
                b_k = (g1(:)'*g1(:)) / (g0(:)'*g0(:) + eps);
            case 'hestenes–stiefe'
                b_k = g1(:)' * (g1(:)-g0(:)) / (dx(:)' * (g1(:)-g0(:)));
            case 'polak-ribiere'
                b_k = g1(:)' * (g1(:)-g0(:)) / (g0(:)'*g0(:));
            otherwise
                error('invalid beta method: %s',args.betaMethod);
        end

        % score the direction to reset instabilities
        restartScore = abs(g1(:)'*g0(:)) / abs(g0(:)'*g0(:));
        if restartScore < 0.1
            b_k=0;
        end

        % update gradient, search direction
        g0 = g1;
        dx = -g1 + b_k*dx;

        % print update
        if ~isempty(args.update_fun)
            time_itr = toc(time_itr); % stop the clock
            args.update_fun(n,cost,x_star,time_itr);
        end

        % set a variable called "exititr" to exit at current iteration
        % when debugging
        if exist('exititr','var')
            break 
        end

    end

end

