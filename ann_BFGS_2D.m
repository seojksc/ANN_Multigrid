%
% Filename : ann_BFGS_2D.m
% two dimensional nonlinear partial differential equation by Finite element method with linear polynomials
% -u'' = 0  in (0, 1)
%  u(0) = 0 = u(1)
%
%
% 2022.01.25
% by J.-K. Seo
%
%
function ann_BFGS_2D(k,H,tol,alpha,w,v)
% k : for mesh size
% H : size of hidden layer
% tol : for convergence rate of optimal control algorithm
% alpha : learning rate for descent gradient iteration
% w, v: for the initial weights (of H)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = k;
h = 1/(N-1); 
x = [ 0:h:(N-1)*h ]'; y = x;     % [h 2h 3h ... Nh]
[xx,yy] = meshgrid(x,y);  % x, y are matrices.
x_ = xx(:);
y_ = yy(:);
clear x y

figure(3); clf;
convergence_rate = 100;
iter_ = 0;
uh = zeros(N,1);

dEdw = zeros(H,3);
dEdv = zeros(H,1);

nu = 1;
dEdF0 = zeros(N^2,1);
dEdF0_ = zeros(N^2,1);
b = 1; % bias node

Bw_inv = 1/60* eye(H*3); % define the initial Bw
Bv_inv = 1/60* eye(H); % define the initial Bv

for j=1:N^2
    x = x_(j);
    y = y_(j);
    uh(j) = x*(x-1)*y*(y-1)*sum(ft_Nxyw(x,y,w,v)) + y*sin(pi*x);
    
    Nxyw = ft_Nxyw(x,y,w,v);
    dNdx = ft_dNdx(x,y,w,v);
    dNdy = ft_dNdy(x,y,w,v);
    d2Ndx2 = ft_d2Ndx2(x,y,w,v);
    d2Ndy2 = ft_d2Ndy2(x,y,w,v);
    dEdF0(j) = ft_dEdF0(nu, x, y, Nxyw, dNdx, dNdy, d2Ndx2, d2Ndy2);
    
    X = w*[x;y;b];
    dF0dw(:,1) = -nu*((2*(y^2-y)+2*(x^2-x))*x*v.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*(v.*grad_sigmoid(X)+x*w(:,1).*v.*grad2_sigmoid(X)) + 2*(x^2-x)*(2*y-1)*x*w(:,2).*v.*grad2_sigmoid(X) + (x^2-x)*(y^2-y)*(2*w(:,1).*v.*grad2_sigmoid(X)+x*w(:,1).^2.*v.*grad3_sigmoid(X)) + (x^2-x)*(y^2-y)*(x*w(:,2).^2.*v.*grad3_sigmoid(X)));
    dF0dw(:,2) = -nu*((2*(y^2-y)+2*(x^2-x))*y*v.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*y*w(:,1).*v.*grad2_sigmoid(X) + 2*(x^2-x)*(2*y-1)*(v.*grad_sigmoid(X)+y*w(:,2).*v.*grad2_sigmoid(X)) + (x^2-x)*(y^2-y)*y*w(:,1).^2.*v.*grad3_sigmoid(X) + (x^2-x)*(y^2-y)*(2*w(:,2).*v.*grad2_sigmoid(X)+y*w(:,2).^2.*v.*grad3_sigmoid(X)));
    dF0dw(:,3) = -nu*((2*(y^2-y)+2*(x^2-x))*b*v.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*(b*w(:,1).*v.*grad2_sigmoid(X)) + 2*(x^2-x)*(2*y-1)*b*w(:,2).*v.*grad2_sigmoid(X) + (x^2-x)*(y^2-y)*(b*w(:,1).^2.*v.*grad3_sigmoid(X)) + (x^2-x)*(y^2-y)*(b*w(:,2).^2.*v.*grad3_sigmoid(X)));
    dEdw = dEdw + dEdF0(j)*dF0dw;
    dF0dv = -nu*((2*(y^2-y)+2*(x^2-x))*sigmoid(X) + 2*(2*x-1)*(y^2-y)*w(:,1).*grad_sigmoid(X) + 2*(x^2-x)*(2*y-1)*w(:,2).*grad_sigmoid(X) + (x^2-x)*(y^2-y)*(w(:,1).^2.*grad2_sigmoid(X)+w(:,2).^2.*grad2_sigmoid(X)));
    dEdv = dEdv + dEdF0(j)*dF0dv;
end  
   f0 = 1/2*sum(dEdF0.^2);
dEdw_old = dEdw;
dEdv_old = dEdv;
Iw = eye(H*3);
Iv = eye(H);

c1 = 10^(-4);
c2 = 0.9;
alpha_ = [0.1,0.5,0.01,0.05,0.001,0.005,0.0001,0.0005,0.00001,0.00005]; 

f0__ = [];
        [M,S] = p1fem(N,1);
A = M+S;
        ut = ft_u(xx(:),yy(:));
    l2_err__ = [];
    h1_err__ = [];

while (convergence_rate > tol)
   gradvalw__ = [];
   w__k = [];
 for kk=1:length(alpha_)
   f0_w = f0 + c1*alpha_(kk)*(dot(-Bw_inv*dEdw_old(:),dEdw_old(:))); % pk = -Bw_inv*dEdw_old(:)   
   delta_w = - alpha_(kk)*Bw_inv*dEdw_old(:);
   w_ = w + reshape(delta_w,H,3);
dEdw_ = zeros(H,3);
   for j=1:N^2
    x = x_(j);
    y = y_(j);
    Nxyw = ft_Nxyw(x,y,w_,v);
    dNdx = ft_dNdx(x,y,w_,v);
    dNdy = ft_dNdy(x,y,w_,v);
    d2Ndx2 = ft_d2Ndx2(x,y,w_,v);
    d2Ndy2 = ft_d2Ndy2(x,y,w_,v);
    dEdF0_(j) = ft_dEdF0(nu, x, y, Nxyw, dNdx, dNdy, d2Ndx2, d2Ndy2);
    
    X = w_*[x;y;b];
    dF0dw(:,1) = -nu*((2*(y^2-y)+2*(x^2-x))*x*v.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*(v.*grad_sigmoid(X)+x*w_(:,1).*v.*grad2_sigmoid(X)) + 2*(x^2-x)*(2*y-1)*x*w_(:,2).*v.*grad2_sigmoid(X) + (x^2-x)*(y^2-y)*(2*w_(:,1).*v.*grad2_sigmoid(X)+x*w_(:,1).^2.*v.*grad3_sigmoid(X)) + (x^2-x)*(y^2-y)*(x*w_(:,2).^2.*v.*grad3_sigmoid(X)));
    dF0dw(:,2) = -nu*((2*(y^2-y)+2*(x^2-x))*y*v.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*y*w_(:,1).*v.*grad2_sigmoid(X) + 2*(x^2-x)*(2*y-1)*(v.*grad_sigmoid(X)+y*w_(:,2).*v.*grad2_sigmoid(X)) + (x^2-x)*(y^2-y)*y*w_(:,1).^2.*v.*grad3_sigmoid(X) + (x^2-x)*(y^2-y)*(2*w_(:,2).*v.*grad2_sigmoid(X)+y*w_(:,2).^2.*v.*grad3_sigmoid(X)));
    dF0dw(:,3) = -nu*((2*(y^2-y)+2*(x^2-x))*b*v.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*(b*w_(:,1).*v.*grad2_sigmoid(X)) + 2*(x^2-x)*(2*y-1)*b*w_(:,2).*v.*grad2_sigmoid(X) + (x^2-x)*(y^2-y)*(b*w_(:,1).^2.*v.*grad3_sigmoid(X)) + (x^2-x)*(y^2-y)*(b*w_(:,2).^2.*v.*grad3_sigmoid(X)));
    dEdw_ = dEdw_ + dEdF0_(j)*dF0dw;
   end 
   f1_w = 1/2*sum(dEdF0_.^2);
   if f1_w <= f0_w
       if -(dot(-Bw_inv*dEdw_old(:),dEdw_(:))) <= -c2*(dot(-Bw_inv*dEdw_old(:),dEdw_old(:)))
          gradvalw__ = [gradvalw__; -(dot(-Bw_inv*dEdw_old(:),dEdw_(:)))];
          w__k = [w__k; kk];
       end
   end
 end
 [~,w_min] = min(gradvalw__);
 alphaw = alpha_(w__k(w_min));
 if isempty(gradvalw__) == 1
     alphaw = alpha;
 end
   gradvalv__ = [];
   v__k = [];
 for kk=1:length(alpha_)
   f0_v = f0 + c1*alpha_(kk)*(dot(-Bv_inv*dEdv_old(:),dEdv_old(:))); % pk = -Bv_inv*dEdv_old(:)   
   delta_v = - alpha_(kk)*Bv_inv*dEdv_old(:);
   v_ = v + delta_v;
dEdv_ = zeros(H,1);
   for j=1:N^2
    x = x_(j);
    y = y_(j);
    Nxyw = ft_Nxyw(x,y,w,v_);
    dNdx = ft_dNdx(x,y,w,v_);
    dNdy = ft_dNdy(x,y,w,v_);
    d2Ndx2 = ft_d2Ndx2(x,y,w,v_);
    d2Ndy2 = ft_d2Ndy2(x,y,w,v_);
    dEdF0_(j) = ft_dEdF0(nu, x, y, Nxyw, dNdx, dNdy, d2Ndx2, d2Ndy2);
    
    X = w*[x;y;b];
    dF0dv = -nu*((2*(y^2-y)+2*(x^2-x))*sigmoid(X) + 2*(2*x-1)*(y^2-y)*w(:,1).*grad_sigmoid(X) + 2*(x^2-x)*(2*y-1)*w(:,2).*grad_sigmoid(X) + (x^2-x)*(y^2-y)*(w(:,1).^2.*grad2_sigmoid(X)+w(:,2).^2.*grad2_sigmoid(X)));
    dEdv_ = dEdv_ + dEdF0_(j)*dF0dv;
   end 
   f1_v = 1/2*sum(dEdF0_.^2);
   if f1_v <= f0_v
       if -(dot(-Bv_inv*dEdv_old(:),dEdv_(:))) <= -c2*(dot(-Bv_inv*dEdv_old(:),dEdv_old(:)))
          gradvalv__ = [gradvalv__; -(dot(-Bv_inv*dEdv_old(:),dEdv_(:)))];
          v__k = [v__k; kk];
       end
   end
 end
 [~,v_min] = min(gradvalv__);
 alphav = alpha_(v__k(v_min));
 if isempty(gradvalv__) == 1
     alphav = alpha;
 end
 
       delta_w = - alphaw*Bw_inv*dEdw_old(:);
       w = w + reshape(delta_w,H,3);
       delta_v = - alphav*Bv_inv*dEdv_old(:);
       v = v + delta_v;
dEdw = zeros(H,3);
dEdv = zeros(H,1);
   for j=1:N^2
    x = x_(j);
    y = y_(j);
    uh(j) = x*(x-1)*y*(y-1)*sum(ft_Nxyw(x,y,w,v)) + y*sin(pi*x);
    
    Nxyw = ft_Nxyw(x,y,w,v);
    dNdx = ft_dNdx(x,y,w,v);
    dNdy = ft_dNdy(x,y,w,v);
    d2Ndx2 = ft_d2Ndx2(x,y,w,v);
    d2Ndy2 = ft_d2Ndy2(x,y,w,v);
    dEdF0(j) = ft_dEdF0(nu, x, y, Nxyw, dNdx, dNdy, d2Ndx2, d2Ndy2);
    
    X = w*[x;y;b];
    dF0dw(:,1) = -nu*((2*(y^2-y)+2*(x^2-x))*x*v.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*(v.*grad_sigmoid(X)+x*w(:,1).*v.*grad2_sigmoid(X)) + 2*(x^2-x)*(2*y-1)*x*w(:,2).*v.*grad2_sigmoid(X) + (x^2-x)*(y^2-y)*(2*w(:,1).*v.*grad2_sigmoid(X)+x*w(:,1).^2.*v.*grad3_sigmoid(X)) + (x^2-x)*(y^2-y)*(x*w(:,2).^2.*v.*grad3_sigmoid(X)));
    dF0dw(:,2) = -nu*((2*(y^2-y)+2*(x^2-x))*y*v.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*y*w(:,1).*v.*grad2_sigmoid(X) + 2*(x^2-x)*(2*y-1)*(v.*grad_sigmoid(X)+y*w(:,2).*v.*grad2_sigmoid(X)) + (x^2-x)*(y^2-y)*y*w(:,1).^2.*v.*grad3_sigmoid(X) + (x^2-x)*(y^2-y)*(2*w(:,2).*v.*grad2_sigmoid(X)+y*w(:,2).^2.*v.*grad3_sigmoid(X)));
    dF0dw(:,3) = -nu*((2*(y^2-y)+2*(x^2-x))*b*v.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*(b*w(:,1).*v.*grad2_sigmoid(X)) + 2*(x^2-x)*(2*y-1)*b*w(:,2).*v.*grad2_sigmoid(X) + (x^2-x)*(y^2-y)*(b*w(:,1).^2.*v.*grad3_sigmoid(X)) + (x^2-x)*(y^2-y)*(b*w(:,2).^2.*v.*grad3_sigmoid(X)));
    dEdw = dEdw + dEdF0(j)*dF0dw;

    dF0dv = -nu*((2*(y^2-y)+2*(x^2-x))*sigmoid(X) + 2*(2*x-1)*(y^2-y)*w(:,1).*grad_sigmoid(X) + 2*(x^2-x)*(2*y-1)*w(:,2).*grad_sigmoid(X) + (x^2-x)*(y^2-y)*(w(:,1).^2.*grad2_sigmoid(X)+w(:,2).^2.*grad2_sigmoid(X)));
    dEdv = dEdv + dEdF0(j)*dF0dv;
   end 
    yw = dEdw(:) - dEdw_old(:);
    dEdw_old = dEdw;
    Bw_inv = (Iw - delta_w*yw'/(yw'*delta_w+eps))*Bw_inv*(Iw - yw*delta_w'/(yw'*delta_w+eps)) + delta_w*delta_w'/(yw'*delta_w+eps); %norm(Bw_inv)

    yv = dEdv - dEdv_old;
    dEdv_old = dEdv;
    Bv_inv = (Iv - delta_v*yv'/(yv'*delta_v+eps))*Bv_inv*(Iv - yv*delta_v'/(yv'*delta_v+eps)) + delta_v*delta_v'/(yv'*delta_v+eps);

    f0 = 1/2*sum(dEdF0.^2);
    f0__= [f0__ f0];
    convergence_rate = f0;
    iter_ = iter_ + 1;
           eh = ut - uh;
        l2_err = sqrt( eh'*M*eh );  h1_err = sqrt( eh'*A*eh );
    l2_err__ = [l2_err__ l2_err];
    h1_err__ = [h1_err__ h1_err];

    if mod(iter_,3) == 0
        ut = ft_u(xx(:),yy(:));
        subplot(161); surf(xx,yy,reshape(ut,N,N));
        subplot(162); surf(xx,yy,reshape(uh,N,N));
        subplot(163); surf(xx,yy,abs(reshape(uh,N,N)-reshape(ut,N,N)));
        axis([0 1 0 1])
        colorbar;
        subplot(164); plot([1:length(f0__)],log(f0__),'b-');
        subplot(165); plot([1:length(f0__)],log10(l2_err__),'b-');
        subplot(166); plot([1:length(f0__)],log10(h1_err__),'b-');

        getframe; % hold on;
save('w','w');
save('v','v');
   fprintf('convergence rate =%12.7g \n',f0);
   iter_

    end

       if iter_ > 25000 
        ut = ft_u(xx(:),yy(:));
       
        eh = ut - uh;
        [M,S] = p1fem(N,1);
        l2_err = sqrt( eh'*M*eh );  h1_err = sqrt( eh'*(M+S)*eh );
        fprintf('Nodes=%4.0f  l2-error=%12.4e H1-error=%12.4e \n',N,l2_err,h1_err);
        fprintf('============================================================\n')
        break; 
       end    
%     f0 = 1/2*sum(dEdF0.^2);
end


ut = ft_u(xx(:),yy(:));
eh = ut - uh;

fprintf(' Total iteration =%12.7g \n',iter_);
[M,S] = p1fem(N,1);
l2_err = sqrt( eh'*M*eh );  h1_err = sqrt( eh'*(M+S)*eh );
fprintf(' Level=%2.0f Nodes=%4.0f  l2-error=%12.4e H1-error=%12.4e \n',k,N,l2_err,h1_err);
fprintf('============================================================\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%
function u = ft_u(x,y)
  u = 1/(exp(pi)-exp(-pi))*sin(pi*x).*(exp(pi*y) - exp(-pi*y));  

function dedf0 = ft_dEdF0(nu, x, y, Nxyw, dNdx, dNdy, d2Ndx2, d2Ndy2)
  dedf0 = -nu*sum((2*(y^2-y)+2*(x^2-x))*Nxyw + 2*(2*x-1)*(y^2-y)*dNdx + 2*(x^2-x)*(2*y-1)*dNdy + (x^2-x)*(y^2-y)*(d2Ndx2+d2Ndy2)) -nu*(-pi^2*y*sin(pi*x)) - ft_F(x,y);

function Nxyw_ = ft_Nxyw(x,y,w,v)
  b = 1;
  Nxyw_ = v.*sigmoid(w*[x;y;b]);

function dNdx_ = ft_dNdx(x,y,w,v)
  b = 1;
  dNdx_ = w(:,1).*v.*grad_sigmoid(w*[x;y;b]);

function dNdy_ = ft_dNdy(x,y,w,v)
  b = 1;
  dNdy_ = w(:,2).*v.*grad_sigmoid(w*[x;y;b]);

function d2Ndx2_ = ft_d2Ndx2(x,y,w,v)
  b = 1;
  d2Ndx2_ = w(:,1).^2.*v.*grad2_sigmoid(w*[x;y;b]);

function d2Ndy2_ = ft_d2Ndy2(x,y,w,v)
  b = 1;
  d2Ndy2_ = w(:,2).^2.*v.*grad2_sigmoid(w*[x;y;b]);

function f_grad3 = grad3_sigmoid(x)
  f_grad3 = grad2_sigmoid(x).*(1-2*sigmoid(x)) - 2*grad_sigmoid(x).^2;

function f_grad2 = grad2_sigmoid(x)
  f_grad2 = grad_sigmoid(x) - 2*sigmoid(x).*grad_sigmoid(x);

function f_grad = grad_sigmoid(x)
  f_grad = sigmoid(x).*(1-sigmoid(x));
%       f_grad = 1;

function ff = sigmoid(x)
  ff = 1./(1+exp(-x));
%       ff = x;

function F = ft_F(x,y)
%  F = -ft_ddu(t)+ft_u2(t);   % -Lap U + U.^2
  F = 0*x + 0*y;   % -Lap U 







