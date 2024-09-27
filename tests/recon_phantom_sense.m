%% Set up the problem
N = 128; % image size
nc = 8; % number of coils
R = 2; % acceleration (undersampling) factor
noise_fac = 0.2; % noise factor

% create the ground truth image
x = phantom(N); % N x N shepp logan phantom

% simulate a sensitivity map
smap = mri_sensemap_sim('nx',N,'ncoil',nc);

% create an (undersampled) spiral sampling pattern
N_samples = ceil(N/R*1e3);
r = linspace(0,pi,N_samples)';
theta = linspace(0,N/R*pi,N_samples)';
omega = r.*[cos(theta),sin(theta)];

% create the system operator
nufft_args = {omega, [N,N], [6,6], 2*[N,N], [N/2,N/2], 'table', 2^10, 'minmax:kb'};
F = Gnufft(true(N), nufft_args); % NUFFT
w = ir_mri_density_comp(omega,'pipe','G',F); % density compensation
FS = Asense(F,smap); % sensitivity encoding

% simulate data (inversely criminal)
y = FS * x;
noise = noise_fac*mean(abs(y(:))) * (rand(N_samples,nc) + 1i*rand(N_samples,nc));
y = y + noise;

%% estimate the initial reconstruction
x0 = FS' * (w.*y); % with density compensation for initialization
x0 = ir_wls_init_scale(FS, y, x0); % fix scale

%% solve with unregularized CG
niter = 20;
slv.CG(x0,FS,y,'niter',niter,...
    'update_fun',@(itr,cost,x_star,time_itr)plot_iteration(112,x,x0,itr,cost,x_star,time_itr));

%% solve with TV-nlCG
niter = 20;
R = reg.TV(1e-3); % spatial total variation (no dim specified)
slv.nlCG(x0,FS,y,'niter',niter,'R',R,...
    'update_fun',@(itr,cost,x_star,time_itr)plot_iteration(112,x,x0,itr,cost,x_star,time_itr));

%% solve with TV-FISTA 
niter = 100;
R = reg.TV(1e-3); % spatial total variation (no dim specified)
[x_star,cost] = slv.FISTA(x0,FS,y,'niter',niter,'R',R,...
    'update_fun',@(itr,cost,x_star,time_itr)plot_iteration(111,x,x0,itr,cost,x_star,time_itr));

%% solve with Wavelet-FISTA
niter = 100;
R = reg.WavL1(1e-3); % spatial total variation (no dim specified)
[x_star,cost] = slv.FISTA(x0,FS,y,'niter',niter,'R',R,...
    'update_fun',@(itr,cost,x_star,time_itr)plot_iteration(111,x,x0,itr,cost,x_star,time_itr));

function plot_iteration(fnum,x,x0,itr,cost,x_star,time_itr)
    
    RMSE_gt0 = sqrt(mean((x(:) - x0(:)).^2));
    RMSE_gt = sqrt(mean((x(:) - x_star(:)).^2));

    figure(fnum)

    subplot(3,1,1)
    imagesc(abs(x0)); axis off
    title(sprintf('Initial estimate\nRMSE = %g',RMSE_gt0));

    subplot(3,1,2)
    imagesc(abs(x_star)); axis off
    title(sprintf('Iteration solution\nRMSE = %g',RMSE_gt));

    subplot(3,1,3)
    plot(0:itr,cost(1:itr+1)); xlabel('iteration #'); ylabel('cost');

    sgtitle(sprintf('Iteration %d, time = %gs',itr,time_itr))

    drawnow

end