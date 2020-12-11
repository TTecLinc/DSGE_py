% 
clear
critvfit = 0.00000000001;
critprice=0.00000001;
bita = 0.95282;
vfitmax=500;
peritmax=500;
theta=0.282;
y=10.0;
rs=1.017;
gam=0.969;
sy=21;
sr=1;
ss=sy*sr;
conaut=zeros(ss,1);
ident=ones(ss,ss);
ptran=load('PIMAT.dat');
shocks=load('YMAT.dat');
shoc=exp(shocks);
%rshoc=rs;
rshoc=rs.*ones(ss,1);
% Calculate utility of permament autarky

for i_ss=1:ss
    if y*shoc(i_ss)<=y*gam;
        conaut(i_ss)=y*shoc(i_ss);
    else
        conaut(i_ss)=y*gam;
    end
    utilaut(i_ss)=utility(conaut(i_ss));
end
utilaut=transpose(utilaut);
temp = ident-bita.*ptran;
tempinv = inv(temp);
vaut = tempinv*utilaut;

ngpb=200;
% Bond Grid 
b=zeros(ngpb,1);
bigg=1;
bmin=-3.30;
bmax = 1.500;
lbb=-3.300;
ubb= -3.30;
fineg=1;
b (1) = bmin;
for i=fineg+1:ngpb
    b (i) = b (i-1) + (bmax-ubb)/((ngpb-fineg-bigg)-1);
    if (b(i)> 0.0 && b(i-1) < 0.0);
        b(i)=0.0;
        izero=i;
    end
end

% Initial Guess the Value
% Policy Function
p=zeros(ngpb,ss);
% Value Function
v=zeros(ngpb,ss);

v00=zeros(ss,1);
for i_ss=1:ss
    p(:,i_ss)=1/rs;
    v(:,i_ss)=vaut(i_ss);
end
i_s_star=zeros(ngpb,ss);
sav=zeros(ngpb,ss);
psav=zeros(ngpb,ss);
utildef=zeros(ss,1);

% PRICE INTERATION : OUTSIDE LOOP 
for kk = 1:vfitmax
    kk
    %  For a given bond price function , compute the optimal value function, by VALUE FUNCTION ITERATION
    for ite = 1:vfitmax;
        vo = v;
        for i_ss=1:ss;
            for i_b=1:ngpb;
                % For given future value , vo , compute optimal consumption and savings 
                assets = b(i_b);
                evo=-1e8;
                i_bp_star = 1;
                for i_bp=1:ngpb;
                    price = p(i_bp, i_ss);
                    c = y*shoc(i_ss) + assets - price*b(i_bp);
                    if c<=0.0;
                        evp=-1e8;
                    else
                        utils=utility(c);
                        evp = 0.0;
                        for i_ssp=1:ss
                            evp = evp + ptran(i_ss,i_ssp)*bita*vo(i_bp,i_ssp);
                        end
                        evp = evp + utils;
                    end
                    if evp>=evo
                        evo = evp;
                        i_bp_star = i_bp;
                    end
                end
                i_s_star(i_b, i_ss) = i_bp_star;
                sav(i_b, i_ss) = b(i_bp_star);
                psav(i_b, i_ss) = p(i_bp_star,i_ss);
                v(i_b, i_ss) = evo;
                savnd=sav;
                psavnd=psav;
            end
        end
        % Compute the value of default 
        % Value of default is a function of value at zero debt :  v_contract(0,s)
        for i_ss=1:ss
            v00(i_ss)=v(izero,i_ss);
        end
        % Value of default is defined recursively as a weighted average of value of default and v_contract(0,s)

        % Vdef=inv(eye(s,s)-beta*(1-theta)*P)*(U+beta*theta*P*v00); 
        
        vbhst = inv(ident-bita*(1-theta)*ptran)*(utilaut+bita.*theta.* ptran*v00);
        
        for i_ss=1:ss
            utildef(i_ss)=utility (conaut(i_ss));
        end
        
        vdef=utildef+bita*ptran*vbhst;
        
        % Now that we have v_contract and v_default (for a given guess of future value (vo) calculate v_option 

        def=zeros(ngpb,ss);
        for i_ss=1:ss
            for i_b=1:ngpb
            if (b(i_b)<0.0) 
                if (v(i_b, i_ss) < vdef(i_ss)) 
                    v(i_b, i_ss) = vdef(i_ss);
                    def(i_b, i_ss) = 1;
                    sav(i_b, i_ss)=0.0;
                    psav(i_b, i_ss)=0.0;
                end
            end
            end
        end

        %----------------------------------------------------------------------------
                
        %  Check whether v_option new = v_option old 
        %  Iterate until convergence 

        gap = max(max(abs (v - vo)));
        %display('gap');
        
        if (gap<critvfit) 
            break
        end
    end

    %  Now we have v_option computed for a given bond price function 

    dprob=zeros(ngpb,ss);
    % Check  whether initial guess on bond price function is consistent with default probabilities
    for i_ss=1:ss
        for i_b=1:ngpb
            evp=0.0;
            for i_ssp=1:ss
                evp = evp + ptran(i_ss,i_ssp)*def(i_b,i_ssp);
            end
            dprob(i_b, i_ss)=evp;
            p1(i_b,i_ss)=(1.0-dprob(i_b,i_ss))/rshoc(i_ss);
        end 
    end

    % Interate on the bond price function until convergence 

    gap = max(max (abs (p1 - p)));
    gap
    if (gap<critprice)
        break
    else
        p=.5*p+.5*p1;
    end
end


% Now we have all the solutions

% Calculate consumption decision rules
for i_ss=1:ss
    for i_b=1:ngpb
        if (def(i_b, i_ss) == 1)
            cons(i_b, i_ss)=conaut(i_ss);
        else
            cons(i_b, i_ss)=y*shoc(i_ss) + b(i_b) - psav(i_b, i_ss)*sav(i_b,i_ss);
        end
        yy(i_b, i_ss)=y*shoc(i_ss);
        consnd(i_b, i_ss)=y*shoc(i_ss) + b(i_b) - psavnd(i_b, i_ss)*savnd(i_b,i_ss);
    end
end
