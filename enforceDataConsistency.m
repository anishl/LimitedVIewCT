clear;%
addpath /n/escanaba/w/anishl/LANL_walnut_comparison/view3dgui

wal_num=103;
lvl_num=4;

model_name = ['v2sgenmsk_lvl' num2str(lvl_num) '_4.1v_ep_8x500[8, 500, 500]_' num2str(wal_num)]
% lvl = model_name(11);
load(['/n/escanaba/w/anishl/Desktop/MSUfiles/results/' ...
     model_name '.1.mat'])
 
dir = ['~/WalnutsOwnReconstructions/Walnut' num2str(wal_num) '/'];
I=fread(fopen([dir 'walnut' num2str(wal_num) '.raw'],'r'),(501^3)*1,'float64');
Z=double(reshape(I,501,501,501*1));
xtrue = Z(:,:,1:501);
xtrue = permute(xtrue,[3,2,1]);% necessary on 2021-02-10 
% if mod(lvl_num,2)==0
%     data_out_normalized = permute(data_out_normalized,[3,2,1]);% necessary on 2021-02-10 
% end

% look into last slice division by zero
data_out_normalized(isnan(data_out_normalized)) = 0;
tmp = zeros(size(data_out_normalized));
% bounding boxes for walnut obsolete as of 2021-06-18
%{
a=1:501;b=a;c=a;

if wal_num == 101
    a = 45:474;
    b = 70:430;
    c = 50:475;
    
    tmp(a,b,c)=data_out_normalized(a,b,c);
    data_out_normalized = tmp;
    clear tmp
end

if wal_num == 103
    c = 100:420;
    b = 95:412;
    a = 60:460;
    
    tmp(a,b,c)=data_out_normalized(a,b,c);
    data_out_normalized = tmp;
    clear tmp
end

if wal_num == 102
    c = 95:425;
    b = 80:400;
    a = 60:445;
    
    tmp(a,b,c)=data_out_normalized(a,b,c);
    data_out_normalized = tmp;
    clear tmp
end

if wal_num == 104
    c = 55:420;
    b = 80:450;
    a = 40:480;
    
    tmp(a,b,c)=data_out_normalized(a,b,c);
    data_out_normalized = tmp;
    clear tmp
end
%}

[a,b,c] = getWalnutBounds(wal_num);
tmp(a,b,c)=data_out_normalized(a,b,c);
data_out_normalized = tmp;
clear tmp
%%
% parameter settings
% nu = 10^5;
nslices = size(data_true,3);
if ~isvar('down'), down = 1; end % down sample a lot to save time
% default is arc detector; but allow flat for cone_beam_ct_example.m
if ~isvar('dfs'), dfs = 0; end

spacing = 5;
n_samples_h_v = 150;
n_lim_view = 4; % must be the same as original limited view fdk recon
dsd = 2000; % source to detector distance
dod = 408; % object detector distance
obt_strt = 22
% limited view cone-beam geometry
cg = ct_geom('fan', 'ns', n_samples_h_v, 'nt', n_samples_h_v, 'na', n_lim_view, ...
'ds', spacing, 'dt', spacing, ...
'down', down, ...
'offset_s', 0.0, ... % quarter detector
'offset_t', 0.0, ...
'dsd', dsd, 'dod', dod, 'dfs', dfs,'orbit_start',obt_strt);


% image geometry
if ~isvar('ig'), printm 'ig: image geometry'
ig = image_geom('nx', size(data_out_normalized,1), 'ny', size(data_out_normalized,2),...
                'nz', nslices, 'fov', ceil(2*cg.rmax), ...
                'down', down);
mask2 = true([ig.nx ig.ny]);
mask2(end) = 0; % trick: test it
ig.mask = repmat(mask2, [1 1 ig.nz]);
clear mask2
end

%%
% LV systm matrices
A_lv = Gcone(cg, ig, 'type', 'sf2', 'nthread', jf('ncore')*2-1);

sino_lv = A_lv*xtrue;

% A_fv = Gcone(cg2, ig, 'type', 'sf2', 'nthread', jf('ncore')*2-1);
%%
lambda = 1;
niter = 50;
[xdc,obj] = cb_data_consistency_ncg_proj(A_lv,sino_lv,data_out_normalized,lambda,a,b,c,niter);

%
dir = ['~/WalnutsOwnReconstructions/Walnut' num2str(wal_num)];
fid=fopen([dir '/walnut' num2str(wal_num) '_lv4.1_' num2str(lvl_num) '.raw'],'w');
fwrite(fid,xdc,'float64');
fclose(fid);

%}

%{
err_lv = (xtrue(:)-data_lv(:))./xtrue(:);err_lv(isnan(err_lv))=0;
err_out = (xtrue(:)-data_out_normalized(:))./xtrue(:);err_out(isnan(err_out))=0;
err_dc = (xtrue(:)-xdc(:))./xtrue(:);err_dc(isnan(err_dc))=0;
mean(abs(err_lv))
mean(abs(err_out))
mean(abs(err_dc))
%}

%
figure();im('mid3',data_out_normalized,[0 0.07]), cbar; colormap gray
figure();im('mid3',data_lv,[0 0.07]), cbar; colormap gray
figure();im('mid3',data_true,[0 0.07]), cbar; colormap gray
figure();im('mid3',xdc,[0 0.07]), cbar; colormap gray
%}


%{
li_full_proc = A_fv*data_out_normalized;
li_lv_acq = A_lv*data_true;
intv = n_full_view/n_lim_view;
ixs = 1:intv:size(li_full_proc,3);

% data consistency
li_full_dc=li_full_proc;
li_full_dc(:,:,ixs) = (li_full_dc(:,:,ixs) + (nu*li_lv_acq))/(1+nu);

%%
% % if ~isvar('xdc'), printm 'fdk on dc_sino'
% % 	xdc = feldkamp(cg2, ig, li_full_dc, ...
% % 		'extrapolate_t', ceil(1.3 * cg2.nt/2)); % todo: compute carefully
% % %		'window', 'hanning,0.7', ... % test window
% % %	clf, im_toggle(xtrue, xfdk, [0 0.02]), return
% % %	im_toggle(permute(xtrue, [1 3 2]), permute(xfdk, [1 3 2]), [0 0.02])
% % prompt
% % end
% % 
% % mae_proc = sum(sum(sum(abs(data_true-data_out_normalized))))/numel(data_true);
% % mae_dc = sum(sum(sum(abs(data_true-xdc))))/numel(data_true);
%}
