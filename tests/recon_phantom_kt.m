% this code is still in progress...

%% Set up the problem
N = 128; % image size
nt = 32; % number of time points
R = 8; % acceleration (undersampling) factor per time point
noise_fac = 0.2; % noise factor

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
omega = r.*[cos(theta),sin(theta)];

% create the system operator
nufft_args = {[N,N], [6,6], 2*[N,N], [N/2,N/2], 'table', 2^10, 'minmax:kb'};
for i = 1:nt
    F = Gnufft(true(N), nufft_args); % NUFFT
    if i == 1
        w = ir_mri_density_comp(omega,'pipe','G',F); % density compensation (for first time point only)
    end
end

% simulate data (inversely criminal)
y = FS * x;
noise = noise_fac*mean(abs(y(:))) * (rand(N_samples,nc) + 1i*rand(N_samples,nc));
y = y + noise;