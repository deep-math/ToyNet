%                                               %
%                                               %
%     Modified version of DeepFool algorithm    %
%                                               %
%                                               %

function adv = mod_adversarial_perturbation(x,l,Df_base,f_out,tn,opts)
    NUM_LABELS = 10;
    OS = 0.02;
    Q = 2;
    MAX_ITER = 50000;
    if(nargin==7)
        if isfield(opts,'labels_limit') NUM_LABELS = opts.labels_limit;end;
        if isfield(opts,'overshoot') OS = opts.overshoot;end;
        if isfield(opts,'norm_p')
            Q = opts.norm_p/(opts.norm_p-1);
            if opts.norm_p==Inf
                Q = 1;
            end
        end
        if isfield(opts,'max_iter') MAX_ITER = opts.max_iter;end;
    end

    Df = @(y,l,idx,tn) Df_base(y,l,idx,tn);
    ff = f_out(x,0,tn);
    ff = ff-ff(l);      % store difference between every classification and classification of interest
    [~,I] = sort(ff,'descend');     % sort in descending order, store indexes of original data placement in I
    labels = I(2:NUM_LABELS);      % get all sorted indexes from 2 to 10

    r = x*0;    % initiate r as 0 vec of size x vec
    x_u = x;

    itr = 0;

    while(f_out(x+(1+OS)*r,1,tn)==l && itr<MAX_ITER)
        itr = itr + 1;

        ff = f_out(x_u,0,tn);
        ff = ff-ff(l);
        idx = [l labels];   % put the label of interest on the first place and sorted indeces follow
        ddf = Df(x_u,l,idx,tn);
        dr = project_boundary_polyhedron(ddf,ff(idx),Q);
        dr = dr/255;
        x_u = x_u + dr;
        r = r + dr;
    end
    disp({'max iter:', itr})
    adv.r = (1+OS)*r;
    adv.new_label = f_out(x+(1+OS)*r,1,tn);
    adv.itr = itr;
end

function dir = project_boundary_polyhedron(Df,f,Q)

res = abs(f)./arrayfun(@(idx) norm(Df(:,idx),Q), 1:size(Df,2));

[~,ii]=min(res);

if isinf(Q)
    dir = res(ii).*(abs(Df(:,ii))>=max(Df(:,ii))).*sign(Df(:,ii));

elseif(Q==1)
    dir = res(ii).*sign(Df(:,ii));

else
    dir = res(ii)*(abs(Df(:,ii))/norm(Df(:,ii),Q)).^(Q-1).*sign(Df(:,ii));
end
end
