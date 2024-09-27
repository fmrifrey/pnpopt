function [x_star, cost] = ADMM(x0, A, b, varargin)

    % set defaults
    defaults.niter = 10; % number of iterations
    defaults.R = []; % regularizers
    defaults.rho = 1.0; % ADMM parameter
    defaults.update_fun = []; % iteration udpate fun

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
            for i = 1:length(args.R)
                c = c + args.R{i}.norm(x);
            end
        elseif ~isempty(args.R)
            c = c + args.R.norm(x);
        end
    end

    % initialize variables
    x_star = x0;
    z = zeros(size(b)); % auxiliary variable
    u = zeros(size(b)); % dual variable
    cost(1) = cost_fun(x_star);

    for n = 1:args.niter
        time_itr = tic; % Start timer

        % primal update (to do)
	    

        % update auxiliary variable
        z = x_star + args.rho*u;
        
        % apply the regularization
        if ~isempty(args.R) && iscell(args.R) % for multiple regularizers
            for j = 1:length(args.R)
                z = args.R{j}.prox(z);
            end
        elseif ~isempty(args.R) % for a single regularizer
            z = args.R.prox(z);
        end
        
        % update dual variable
        u = u + x_star - z;

        % calculate the cost function
        time_cost = tic; % time the cost calculation
        cost(n+1) = cost_fun(x_star);
        time_itr = time_itr - toc(time_cost); % remove cost from total time

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