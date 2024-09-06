function [x_star,cost] = CG(x0, A, b, varargin)

    % set defaults
    defaults.niter = 5; % number of iterations
    defaults.R = []; % regularizers
    defaults.talk2me = 1; % option to print update messages
    
    % parse inputs
    args = vararg_pair(defaults,varargin);

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
    r = reshape(A'*(b - A*x_star), size(x_star)); % residual
    p = r; % initial search direction = residual
    cost(1) = cost_fun(x_star);

    % print update
    if args.talk2me
        fprintf('CG initialization, cost = %g\n', cost(1));
    end

    for n = 1:args.niter
        time_itr = tic; % start timer

        % calculate the gradient descent step
        AtAdx = reshape(A'*(A*p), size(x0));
        alpha = (r(:)' * r(:)) / (p(:)' * AtAdx(:));
        
        % update the solution x*
        x_star = x_star + alpha * p;

        % apply the regularization
        if ~isempty(args.R) && iscell(args.R) % for multiple regularizers
            for j = 1:length(args.R)
                x_star = args.R{j}.prox(x_star);
            end
        elseif ~isempty(args.R) % for a single regularizer
            x_star = args.R.prox(x_star);
        end

        % calculate new residual & update search direction
        r_new = r - alpha * AtAdx;
        beta = (r_new(:)' * r_new(:)) / (r(:)' * r(:));
        r = r_new;
        p = r + beta * p;

        % calculate the cost function
        time_cost = tic; % time the cost calculation
        cost(n+1) = cost_fun(x_star);
        time_itr = time_itr - toc(time_cost); % remove cost from total time

        % print update
        if args.talk2me
            time_itr = toc(time_itr); % stop the clock
            fprintf('CG iteration %d/%d', n, args.niter);
            fprintf(', cost = %g', cost(n+1));
            fprintf(', iteration time = %.3fs\n', time_itr);
        end

        % set a variable called "exititr" to exit at current iteration
        % when debugging
        if exist('exititr','var')
            break 
        end

    end
    
end