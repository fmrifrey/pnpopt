% THIS CODE IS STILL IN PROGRESS!!

function [x_star, cost] = ADMM(x0, A, b, varargin)

    % set defaults
    defaults.niter = 10; % number of iterations
    defaults.R = []; % regularizers
    defaults.rho = 1.0; % ADMM parameter
    defaults.talk2me = 1; % option to print update messages

    % parse inputs
    args = vararg_pair(defaults, varargin);

    % initialize cost
    cost = zeros(1,args.niter+1);

    % define the cost function
    function c = cost_fun(x)
        % calculate data fidelity term
        c = 0.5 * norm(A * x - b, 2)^2;

        % add regularization norms
        if ~isempty(args.R) && iscell(args.R)
            for i = 1:length(R)
                c = c + args.R{i}.lam * args.R{i}.norm(x);
            end
        elseif ~isempty(args.R)
            c = c + args.R.lam * args.R.norm(x);
        end
    end

    % initialize variables
    x_star = x0;
    z = zeros(size(x0)); % auxiliary variable
    u = zeros(size(x0)); % dual variable
    cost(1) = cost_fun(x_star);

    for iter = 1:args.niter
        time_itr = tic; % Start timer

        % update x_star (primal variable)
        if ismatrix(A) % gram can be calculated explicitly
            x_star = (A' * A + args.rho * eye(size(A, 2))) \ ...
                (A' * b + args.rho * (z - u));
        else % gram must be solved for with CG

        end

        % update auxiliary variable
        if ~isempty(args.prox_g)
            z = args.prox_g(x_star + u);
        end

        % update dual variable
        u = u + x_star - z;

        % calculate the cost function
        time_cost = tic; % time the cost calculation
        cost(n+1) = cost_fun(x_star);
        time_itr = time_itr - toc(time_cost); % remove cost from total time

        % print update
        if args.talk2me
            time_itr = toc(time_itr); % stop the clock
            fprintf('ADMM iteration %d/%d', iter, args.niter);
            fprintf(', cost = %g', cost(n+1));
            fprintf(', iteration time = %.3fs\n', time_itr);
        end

    end
    
end