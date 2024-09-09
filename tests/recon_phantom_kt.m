%% Set up the problem
N = 128; % image size
nt = 32; % number of time points
R = 2; % acceleration (undersampling) factor per time point
noise_fac = 0.3; % noise factor

% create the dynamic ground truth image
[x_frame1,e] = phantom(N); % N x N shepp logan phantom
x = zeros(N,N,nt); % N x N x nt dynamic image

% create random smooth movement of the left ventricle
mvmt_a = 0.25*rand(1,5);
mvmt_f = 0.15*rand(1,5);
for i = 1:nt
    ei = e;
    ei(4,2:6) = ei(4,2:6).*(mvmt_a.*sin(2*pi*mvmt_f*i)+1);
    x(:,:,i) = phantom(N,ei);
end

%% create an (undersampled) spiral sampling pattern
N_samples = ceil(N/R*1e3);
r = linspace(0,pi,N_samples)';
theta = linspace(0,N/R*pi,N_samples)';
omega_0 = r.*[cos(theta),sin(theta)];

% create the dynamic system operators
nufft_args = {[N,N], [6,6], 2*[N,N], [N/2,N/2], 'table', 2^10, 'minmax:kb'};
F_set = cell(nt,1);
for i = 1:nt
    theta = (i-1)*(3-sqrt(5))*pi/2;
    rot_i = [cos(theta),sin(theta);-sin(theta),cos(theta)];
    omega_i = omega_0*rot_i';
    F_set{i} = Gnufft(true(N), [omega_i,nufft_args]); % NUFFT
end
w = ir_mri_density_comp(omega_0,'pipe','G',F_set{1}); % density compensation (for first time point only)
F_block = block_fatrix(F_set);

% create a fatrix that reshapes the dynamic data to make things easier:
F = fatrix2('idim', [N,N,nt], ...
    'odim', [N_samples,nt], ...
    'does_many', 1, ...
    'forw', @(~,x)reshape(F_block*x(:),[N_samples,nt]), ...
    'back', @(~,y)reshape(F_block'*y(:),[N,N,nt]));

%% simulate data (inversely criminal)
y = F * x;
noise = noise_fac*mean(abs(y(:))) * (rand(N_samples,nt) + 1i*rand(N_samples,nt));
y = y + noise;

%% estimate the initial reconstruction
x0 = F' * (w.*y); % with density compensation for initialization
x0 = ir_wls_init_scale(F, y, x0); % fix scale

%% solve with unregularized CG
niter = 20;
slv.CG(x0,F,y,'niter',niter,...
    'update_fun',@(itr,cost,x_star,time_itr)plot_iteration(112,x,x0,itr,cost,x_star,time_itr));

%% solve with TV-FISTA 
niter = 30;
R = reg.TV([1e-2,1e-2,0.1]); % spatial total variation (no dim specified)
x_star = slv.FISTA(x0,F,y,'niter',niter,'R',R,...
    'update_fun',@(itr,cost,x_star,time_itr)plot_iteration(111,x,x0,itr,cost,x_star,time_itr));

function plot_iteration(fnum,x,x0,itr,cost,x_star,time_itr)
    
    RMSE_gt0 = sqrt(mean((x(:) - x0(:)).^2));
    RMSE_gt = sqrt(mean((x(:) - x_star(:)).^2));

    figure(fnum)

    subplot(3,1,1)
    imagesc(reshape(abs(x0(:,:,1:4)),size(x0,1),[])); axis off
    title(sprintf('Initial estimate (first 4 frames)\nRMSE = %g',RMSE_gt0));

    subplot(3,1,2)
    imagesc(reshape(abs(x_star(:,:,1:4)),size(x0,1),[])); axis off
    title(sprintf('Iteration solution (first 4 frames)\nRMSE = %g',RMSE_gt));

    subplot(3,1,3)
    plot(0:itr,cost(1:itr+1)); xlabel('iteration #'); ylabel('cost');

    sgtitle(sprintf('Iteration %d, time = %gs',itr,time_itr))

    drawnow

end
