% 
% Filename : ann_BFGS_multigrid_2D.m
% two dimensional nonlinear partial differential equation by Finite element method with linear polynomials
% -u'' = 0  in (0, 1)
%  u(0) = 0 = u(1)
%
%
% 2022.02.25.(Fri)
%
% by J.-K. Seo
%
% 
%
function ann_BFGS_multigrid_2D(k,H,tol,alpha,w1,v1)
% k : for mesh size
% H : size of hidden layer on find-grid
% tol : for convergence rate of optimal control algorithm
% alpha : learning rate for descent gradient iteration
% w1, v1: for the initial weights on fine-grid (of H)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = k;
h = 1/(N-1); 
x = [ 0:h:(N-1)*h ]'; y = x;     % [h 2h 3h ... Nh]
[xx,yy] = meshgrid(x,y);  % x, y are matrices.
x_ = xx(:);
y_ = yy(:);
x_2 = xx(3:2:end-2,3:2:end-2);
y_2 = yy(3:2:end-2,3:2:end-2);
x_2 = x_2(:);
y_2 = y_2(:);
clear x y

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N1 = size(xx, 1);

x1 = -1; x2 = -1;
y1 = -1; y2 = -1;
xy  = 4;
%        y2         | J1 = diag(y1, xy, y2);  J2 = diag(x1, 0, x2)
%  x1    xy   x2    | => J1*U + U*J2 = h^2*F
%        y1         | =>  A*uh = Fh  
%               where  A=kron(eye(N),J1)+kron(J2,eye(N)),  Fh = h^2*F(:)     
% In Matlab, tensor product tensor(A,B) = kron(B,A) 
J1 = sparse(N1,N1); J2 = J1; A1 = sparse(N1^2,N1^2);
J1 = diag(xy*ones(N1,1)) + diag(y1*ones(N1-1,1),-1) + diag(y2*ones(N1-1,1),1);
J2 = diag(x1*ones(N1-1,1),-1) + diag(x2*ones(N1-1,1),1);
A1 = kron(J2, speye(N1)) + kron(speye(N1), J1);
A1 = A1/(h)^2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(2); clf;
convergence_rate = 100;
f0__= [];
iter = 0;
iter_ = 0;
uh = zeros(N,1);

%==========================================================================
H2 = 12;
        w0 = load('w0_48');
        v0 = load('v0_48');
        w_ = w0.w;
        v_ = v0.v;
w2 = w_(1:12,:);
v2 = sqrt(48)/sqrt(12)*v_(1:12,:);
%==========================================================================
nu = 1;

dEdF0__ = zeros(N^2,1);
b = 1; % bias node

u_xx_yy = zeros(N^2,1);

proLap = zeros(N,N);

   for j=1:N^2
    x = x_(j);
    y = y_(j);
    uh(j) = x*(x-1)*y*(y-1)*sum(ft_Nxyw(x,y,w1,v1)) + y*sin(pi*x);
   end
eN = zeros(N^2,1);

dEdF0_ = zeros(N^2,1);
Bw_inv = 1/60* eye(H*3); % define the initial Bw
Bv_inv = 1/60* eye(H); % define the initial Bv
Bw2_inv = 1/60* eye(H2*3); % define the initial Bw
Bv2_inv = 1/60* eye(H2); % define the initial Bv
c1 = 10^(-4);
c2 = 0.9;
alpha_ = [0.1,0.5,0.01,0.05,0.001,0.005,0.0001,0.0005,0.00001,0.00005]; 
Iw = eye(H*3);
Iv = eye(H);
Iw2 = eye(H2*3);
Iv2 = eye(H2);

co_num = 1;
fi_num = 3;
beta = 1;

[M,S] = p1fem(N,1);
A = M+S;
        ut = ft_u(xx(:),yy(:));
    l2_err__ = [];
    h1_err__ = [];
    
dEdw = zeros(H,3);
dEdv = zeros(H,1);
dEdF0 = zeros(N^2,1);
   for j=1:N^2
    x = x_(j);
    y = y_(j);
    
    Nxyw_1 = ft_Nxyw(x,y,w1,v1);
    dNdx_1 = ft_dNdx(x,y,w1,v1);
    dNdy_1 = ft_dNdy(x,y,w1,v1);
    d2Ndx2_1 = ft_d2Ndx2(x,y,w1,v1);
    d2Ndy2_1 = ft_d2Ndy2(x,y,w1,v1);
    if iter < 1000 
    dEdF0(j) = ft_dudxy(nu, x, y, Nxyw_1, dNdx_1, dNdy_1, d2Ndx2_1, d2Ndy2_1) -nu*(-pi^2*y*sin(pi*x))   - ft_F(x,y); % Laplacian without boundary data
    else
    dEdF0(j) = ft_dudxy(nu, x, y, Nxyw_1, dNdx_1, dNdy_1, d2Ndx2_1, d2Ndy2_1) -nu*(-pi^2*y*sin(pi*x)) + beta*d2u_dx2_dy2_(j)  - ft_F(x,y); % Laplacian without boundary data
    end
    X = w1*[x;y;b];
    dF0dw(:,1) = -nu*((2*(y^2-y)+2*(x^2-x))*x*v1.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*(v1.*grad_sigmoid(X)+x*w1(:,1).*v1.*grad2_sigmoid(X)) + 2*(x^2-x)*(2*y-1)*x*w1(:,2).*v1.*grad2_sigmoid(X) + (x^2-x)*(y^2-y)*(2*w1(:,1).*v1.*grad2_sigmoid(X)+x*w1(:,1).^2.*v1.*grad3_sigmoid(X)) + (x^2-x)*(y^2-y)*(x*w1(:,2).^2.*v1.*grad3_sigmoid(X)));
    dF0dw(:,2) = -nu*((2*(y^2-y)+2*(x^2-x))*y*v1.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*y*w1(:,1).*v1.*grad2_sigmoid(X) + 2*(x^2-x)*(2*y-1)*(v1.*grad_sigmoid(X)+y*w1(:,2).*v1.*grad2_sigmoid(X)) + (x^2-x)*(y^2-y)*y*w1(:,1).^2.*v1.*grad3_sigmoid(X) + (x^2-x)*(y^2-y)*(2*w1(:,2).*v1.*grad2_sigmoid(X)+y*w1(:,2).^2.*v1.*grad3_sigmoid(X)));
    dF0dw(:,3) = -nu*((2*(y^2-y)+2*(x^2-x))*b*v1.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*(b*w1(:,1).*v1.*grad2_sigmoid(X)) + 2*(x^2-x)*(2*y-1)*b*w1(:,2).*v1.*grad2_sigmoid(X) + (x^2-x)*(y^2-y)*(b*w1(:,1).^2.*v1.*grad3_sigmoid(X)) + (x^2-x)*(y^2-y)*(b*w1(:,2).^2.*v1.*grad3_sigmoid(X)));
    dEdw = dEdw + dEdF0(j)*dF0dw;

    dF0dv = -nu*((2*(y^2-y)+2*(x^2-x))*sigmoid(X) + 2*(2*x-1)*(y^2-y)*w1(:,1).*grad_sigmoid(X) + 2*(x^2-x)*(2*y-1)*w1(:,2).*grad_sigmoid(X) + (x^2-x)*(y^2-y)*(w1(:,1).^2.*grad2_sigmoid(X)+w1(:,2).^2.*grad2_sigmoid(X)));
    dEdv = dEdv + dEdF0(j)*dF0dv;
   end 
f0 = 1/2*sum(dEdF0.^2);
dEdw_old = dEdw;
dEdv_old = dEdv;

while (convergence_rate > tol)
if iter~=0
   for j=1:N^2
    x = x_(j);
    y = y_(j);

    Nxyw = ft_Nxyw(x,y,w1,v1);
    dNdx = ft_dNdx(x,y,w1,v1);
    dNdy = ft_dNdy(x,y,w1,v1);
    d2Ndx2 = ft_d2Ndx2(x,y,w1,v1);
    d2Ndy2 = ft_d2Ndy2(x,y,w1,v1);
    u_xx_yy(j) = ft_dudxy(nu, x, y, Nxyw, dNdx, dNdy, d2Ndx2, d2Ndy2) -nu*(-pi^2*y*sin(pi*x)); % Laplacian of u
    eN(j) = u_xx_yy(j) - ft_F(x_(j),y_(j));
   end
   
       eN2 = reshape(eN,N,N); % define the restriction function of eN
       for ii=3:2:N-2
           for jj=3:2:N-2
               eN2(ii,jj) = 1/8*(eN2(ii,jj-1) + eN2(ii,jj+1) + 2*eN2(ii,jj) + eN2(ii-1,jj) + eN2(ii+1,jj))...
                         + 1/16*(eN2(ii-1,jj-1) + eN2(ii-1,jj+1) + eN2(ii+1,jj-1) + eN2(ii+1,jj+1));
           end 
       end 
       eN2 = eN2(3:2:N-2,3:2:N-2);
       [size_eN2_x,size_eN2_y] = size(eN2);
       eN2 = eN2(:);
       x2_ = x_2;
       y2_ = y_2;
       
%==========================================================================       
dEdw2 = zeros(H2,3);
dEdv2 = zeros(H2,1);
dEdF00 = zeros(size_eN2_x*size_eN2_y,1);
   for j=1:size_eN2_x*size_eN2_y
    x = x2_(j);
    y = y2_(j);
    
    Nxyw = ft_Nxyw(x,y,w2,v2);
    dNdx = ft_dNdx(x,y,w2,v2);
    dNdy = ft_dNdy(x,y,w2,v2);
    d2Ndx2 = ft_d2Ndx2(x,y,w2,v2);
    d2Ndy2 = ft_d2Ndy2(x,y,w2,v2);
    dEdF00(j) = ft_dudxy(nu, x, y, Nxyw, dNdx, dNdy, d2Ndx2, d2Ndy2)  - eN2(j); % Laplacian without boundary data
    
    X = w2*[x;y;b];
    dF0dw2(:,1) = -nu*((2*(y^2-y)+2*(x^2-x))*x*v2.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*(v2.*grad_sigmoid(X)+x*w2(:,1).*v2.*grad2_sigmoid(X)) + 2*(x^2-x)*(2*y-1)*x*w2(:,2).*v2.*grad2_sigmoid(X) + (x^2-x)*(y^2-y)*(2*w2(:,1).*v2.*grad2_sigmoid(X)+x*w2(:,1).^2.*v2.*grad3_sigmoid(X)) + (x^2-x)*(y^2-y)*(x*w2(:,2).^2.*v2.*grad3_sigmoid(X)));
    dF0dw2(:,2) = -nu*((2*(y^2-y)+2*(x^2-x))*y*v2.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*y*w2(:,1).*v2.*grad2_sigmoid(X) + 2*(x^2-x)*(2*y-1)*(v2.*grad_sigmoid(X)+y*w2(:,2).*v2.*grad2_sigmoid(X)) + (x^2-x)*(y^2-y)*y*w2(:,1).^2.*v2.*grad3_sigmoid(X) + (x^2-x)*(y^2-y)*(2*w2(:,2).*v2.*grad2_sigmoid(X)+y*w2(:,2).^2.*v2.*grad3_sigmoid(X)));
    dF0dw2(:,3) = -nu*((2*(y^2-y)+2*(x^2-x))*b*v2.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*(b*w2(:,1).*v2.*grad2_sigmoid(X)) + 2*(x^2-x)*(2*y-1)*b*w2(:,2).*v2.*grad2_sigmoid(X) + (x^2-x)*(y^2-y)*(b*w2(:,1).^2.*v2.*grad3_sigmoid(X)) + (x^2-x)*(y^2-y)*(b*w2(:,2).^2.*v2.*grad3_sigmoid(X)));
    dEdw2 = dEdw2 + dEdF00(j)*dF0dw2; 

    dF0dv = -nu*((2*(y^2-y)+2*(x^2-x))*sigmoid(X) + 2*(2*x-1)*(y^2-y)*w2(:,1).*grad_sigmoid(X) + 2*(x^2-x)*(2*y-1)*w2(:,2).*grad_sigmoid(X) + (x^2-x)*(y^2-y)*(w2(:,1).^2.*grad2_sigmoid(X)+w2(:,2).^2.*grad2_sigmoid(X)));
    dEdv2 = dEdv2 + dEdF00(j)*dF0dv;
   end 
f0_ = 1/2*sum(dEdF00.^2);
dEdw_old2 = dEdw2;
dEdv_old2 = dEdv2;

dEdF0_ = zeros(size_eN2_x*size_eN2_y,1);
for kkkkk=1:co_num
   gradvalw__ = [];
   w__k = [];
 for kk=1:length(alpha_)
   f0_w2 = f0_ + c1*alpha_(kk)*(dot(-Bw2_inv*dEdw_old2(:),dEdw_old2(:))); % pk = -Bw_inv*dEdw_old(:)   
   delta_w = - alpha_(kk)*Bw2_inv*dEdw_old2(:);
   w_ = w2 + reshape(delta_w,H2,3);
dEdw_ = zeros(H2,3);
   for j=1:size_eN2_x*size_eN2_y
    x = x2_(j);
    y = y2_(j);
    Nxyw = ft_Nxyw(x,y,w_,v2);
    dNdx = ft_dNdx(x,y,w_,v2);
    dNdy = ft_dNdy(x,y,w_,v2);
    d2Ndx2 = ft_d2Ndx2(x,y,w_,v2);
    d2Ndy2 = ft_d2Ndy2(x,y,w_,v2);
    dEdF0_(j) = ft_dudxy(nu, x, y, Nxyw, dNdx, dNdy, d2Ndx2, d2Ndy2)  - eN2(j); % Laplacian without boundary data
                
    X = w_*[x;y;b];
    dF0dw2(:,1) = -nu*((2*(y^2-y)+2*(x^2-x))*x*v2.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*(v2.*grad_sigmoid(X)+x*w_(:,1).*v2.*grad2_sigmoid(X)) + 2*(x^2-x)*(2*y-1)*x*w_(:,2).*v2.*grad2_sigmoid(X) + (x^2-x)*(y^2-y)*(2*w_(:,1).*v2.*grad2_sigmoid(X)+x*w_(:,1).^2.*v2.*grad3_sigmoid(X)) + (x^2-x)*(y^2-y)*(x*w_(:,2).^2.*v2.*grad3_sigmoid(X)));
    dF0dw2(:,2) = -nu*((2*(y^2-y)+2*(x^2-x))*y*v2.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*y*w_(:,1).*v2.*grad2_sigmoid(X) + 2*(x^2-x)*(2*y-1)*(v2.*grad_sigmoid(X)+y*w_(:,2).*v2.*grad2_sigmoid(X)) + (x^2-x)*(y^2-y)*y*w_(:,1).^2.*v2.*grad3_sigmoid(X) + (x^2-x)*(y^2-y)*(2*w_(:,2).*v2.*grad2_sigmoid(X)+y*w_(:,2).^2.*v2.*grad3_sigmoid(X)));
    dF0dw2(:,3) = -nu*((2*(y^2-y)+2*(x^2-x))*b*v2.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*(b*w_(:,1).*v2.*grad2_sigmoid(X)) + 2*(x^2-x)*(2*y-1)*b*w_(:,2).*v2.*grad2_sigmoid(X) + (x^2-x)*(y^2-y)*(b*w_(:,1).^2.*v2.*grad3_sigmoid(X)) + (x^2-x)*(y^2-y)*(b*w_(:,2).^2.*v2.*grad3_sigmoid(X)));
    dEdw_ = dEdw_ + dEdF0_(j)*dF0dw2;
   end 
   f1_w = 1/2*sum(dEdF0_.^2);
   if f1_w <= f0_w2
       if -(dot(-Bw2_inv*dEdw_old2(:),dEdw_(:))) <= -c2*(dot(-Bw2_inv*dEdw_old2(:),dEdw_old2(:)))
          gradvalw__ = [gradvalw__; -(dot(-Bw2_inv*dEdw_old2(:),dEdw_(:)))];
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
   f0_v = f0_ + c1*alpha_(kk)*(dot(-Bv2_inv*dEdv_old2(:),dEdv_old2(:))); % pk = -Bv_inv*dEdv_old(:)   
   delta_v = - alpha_(kk)*Bv2_inv*dEdv_old2(:);
   v_ = v2 + delta_v;
dEdv_ = zeros(H2,1);
   for j=1:size_eN2_x*size_eN2_y
    x = x2_(j);
    y = y2_(j);
    Nxyw = ft_Nxyw(x,y,w2,v_);
    dNdx = ft_dNdx(x,y,w2,v_);
    dNdy = ft_dNdy(x,y,w2,v_);
    d2Ndx2 = ft_d2Ndx2(x,y,w2,v_);
    d2Ndy2 = ft_d2Ndy2(x,y,w2,v_);
    dEdF0_(j) = ft_dudxy(nu, x, y, Nxyw, dNdx, dNdy, d2Ndx2, d2Ndy2)  - eN2(j);
                
    X = w2*[x;y;b];
    dF0dv = -nu*((2*(y^2-y)+2*(x^2-x))*sigmoid(X) + 2*(2*x-1)*(y^2-y)*w2(:,1).*grad_sigmoid(X) + 2*(x^2-x)*(2*y-1)*w2(:,2).*grad_sigmoid(X) + (x^2-x)*(y^2-y)*(w2(:,1).^2.*grad2_sigmoid(X)+w2(:,2).^2.*grad2_sigmoid(X)));
    dEdv_ = dEdv_ + dEdF0_(j)*dF0dv;
   end 
   f1_v = 1/2*sum(dEdF0_.^2);
   if f1_v <= f0_v
       if -(dot(-Bv2_inv*dEdv_old2(:),dEdv_(:))) <= -c2*(dot(-Bv2_inv*dEdv_old2(:),dEdv_old2(:)))
          gradvalv__ = [gradvalv__; -(dot(-Bv2_inv*dEdv_old2(:),dEdv_(:)))];
          v__k = [v__k; kk];
       end
   end
 end
 [~,v_min] = min(gradvalv__);
 alphav = alpha_(v__k(v_min));
 if isempty(gradvalv__) == 1
     alphav = alpha;
 end
 
       delta_w = - alphaw*Bw2_inv*dEdw_old2(:);
       w2 = w2 + reshape(delta_w,H2,3);
       delta_v = - alphav*Bv2_inv*dEdv_old2(:);
       v2 = v2 + delta_v;
dEdw2 = zeros(H2,3);
dEdv2 = zeros(H2,1);
   for j=1:size_eN2_x*size_eN2_y
    x = x2_(j);
    y = y2_(j);
        
    Nxyw = ft_Nxyw(x,y,w2,v2);
    dNdx = ft_dNdx(x,y,w2,v2);
    dNdy = ft_dNdy(x,y,w2,v2);
    d2Ndx2 = ft_d2Ndx2(x,y,w2,v2);
    d2Ndy2 = ft_d2Ndy2(x,y,w2,v2);
    dEdF00(j) = ft_dudxy(nu, x, y, Nxyw, dNdx, dNdy, d2Ndx2, d2Ndy2)  -nu*(-pi^2*y*sin(pi*x)) - eN2(j); % Laplacian without boundary data
    
    X = w2*[x;y;b];
    dF0dw2(:,1) = -nu*((2*(y^2-y)+2*(x^2-x))*x*v2.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*(v2.*grad_sigmoid(X)+x*w2(:,1).*v2.*grad2_sigmoid(X)) + 2*(x^2-x)*(2*y-1)*x*w2(:,2).*v2.*grad2_sigmoid(X) + (x^2-x)*(y^2-y)*(2*w2(:,1).*v2.*grad2_sigmoid(X)+x*w2(:,1).^2.*v2.*grad3_sigmoid(X)) + (x^2-x)*(y^2-y)*(x*w2(:,2).^2.*v2.*grad3_sigmoid(X)));
    dF0dw2(:,2) = -nu*((2*(y^2-y)+2*(x^2-x))*y*v2.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*y*w2(:,1).*v2.*grad2_sigmoid(X) + 2*(x^2-x)*(2*y-1)*(v2.*grad_sigmoid(X)+y*w2(:,2).*v2.*grad2_sigmoid(X)) + (x^2-x)*(y^2-y)*y*w2(:,1).^2.*v2.*grad3_sigmoid(X) + (x^2-x)*(y^2-y)*(2*w2(:,2).*v2.*grad2_sigmoid(X)+y*w2(:,2).^2.*v2.*grad3_sigmoid(X)));
    dF0dw2(:,3) = -nu*((2*(y^2-y)+2*(x^2-x))*b*v2.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*(b*w2(:,1).*v2.*grad2_sigmoid(X)) + 2*(x^2-x)*(2*y-1)*b*w2(:,2).*v2.*grad2_sigmoid(X) + (x^2-x)*(y^2-y)*(b*w2(:,1).^2.*v2.*grad3_sigmoid(X)) + (x^2-x)*(y^2-y)*(b*w2(:,2).^2.*v2.*grad3_sigmoid(X)));
    dEdw2 = dEdw2 + dEdF00(j)*dF0dw2;

    dF0dv = -nu*((2*(y^2-y)+2*(x^2-x))*sigmoid(X) + 2*(2*x-1)*(y^2-y)*w2(:,1).*grad_sigmoid(X) + 2*(x^2-x)*(2*y-1)*w2(:,2).*grad_sigmoid(X) + (x^2-x)*(y^2-y)*(w2(:,1).^2.*grad2_sigmoid(X)+w2(:,2).^2.*grad2_sigmoid(X)));
    dEdv2 = dEdv2 + dEdF00(j)*dF0dv;

   end 
    yw = dEdw2(:) - dEdw_old2(:);
    dEdw_old2 = dEdw2;
    Bw2_inv = (Iw2 - delta_w*yw'/(yw'*delta_w+eps))*Bw2_inv*(Iw2 - yw*delta_w'/(yw'*delta_w+eps)) + delta_w*delta_w'/(yw'*delta_w+eps); %norm(Bw_inv)

    yv = dEdv2 - dEdv_old2;
    dEdv_old2 = dEdv2;
    Bv2_inv = (Iv2 - delta_v*yv'/(yv'*delta_v+eps))*Bv2_inv*(Iv2 - yv*delta_v'/(yv'*delta_v+eps)) + delta_v*delta_v'/(yv'*delta_v+eps);

    f0_ = 1/2*sum(dEdF00.^2);
end

end
%==========================================================================
   uh_ = zeros(N^2,1);
   d2u_dx2_dy2_ = zeros(N^2,1);
   for j=1:N^2
    x = x_(j);
    y = y_(j);
    uh_(j) = (x*(x-1)*y*(y-1)*sum(ft_Nxyw(x,y,w2,v2)));% + y*sin(pi*x));

    Nxyw = ft_Nxyw(x,y,w2,v2);
    dNdx = ft_dNdx(x,y,w2,v2);
    dNdy = ft_dNdy(x,y,w2,v2);
    d2Ndx2 = ft_d2Ndx2(x,y,w2,v2);
    d2Ndy2 = ft_d2Ndy2(x,y,w2,v2);
    d2u_dx2_dy2_(j) = ft_dudxy(nu, x, y, Nxyw, dNdx, dNdy, d2Ndx2, d2Ndy2) ; % Laplacian of u
   end

       Lap_2 = reshape(d2u_dx2_dy2_,N,N); % define the restriction function of eN
       for ii=3:2:N-2
           for jj=3:2:N-2
               proLap(ii,jj) = 1/2*Lap_2(ii,jj);
               proLap(ii+1,jj) = 1/4*(Lap_2(ii,jj) + Lap_2(ii+2,jj));
               proLap(ii,jj+1) = 1/4*(Lap_2(ii,jj) + Lap_2(ii,jj+2));
               proLap(ii+1,jj+1) = 1/8*(Lap_2(ii,jj) + Lap_2(ii+2,jj) + Lap_2(ii,jj+2) + Lap_2(ii+2,jj+2));
           end 
       end
       d2u_dx2_dy2_ = proLap;
   
%==========================================================================

beta = 1;

for kkkkkk=1:fi_num
   gradvalw__ = [];
   w__k = [];
 for kk=1:length(alpha_)
   f0_w = f0 + c1*alpha_(kk)*(dot(-Bw_inv*dEdw_old(:),dEdw_old(:))); % pk = -Bw_inv*dEdw_old(:)   
   delta_w = - alpha_(kk)*Bw_inv*dEdw_old(:);
   w_ = w1 + reshape(delta_w,H,3);
dEdw_ = zeros(H,3);
   for j=1:N^2
    x = x_(j);
    y = y_(j);
    Nxyw = ft_Nxyw(x,y,w_,v1);
    dNdx = ft_dNdx(x,y,w_,v1);
    dNdy = ft_dNdy(x,y,w_,v1);
    d2Ndx2 = ft_d2Ndx2(x,y,w_,v1);
    d2Ndy2 = ft_d2Ndy2(x,y,w_,v1);
    
        
    if mod(iter,2) == 1 
    dEdF0__(j) = ft_dudxy(nu, x, y, Nxyw, dNdx, dNdy, d2Ndx2, d2Ndy2) -nu*(-pi^2*y*sin(pi*x))   - ft_F(x,y);
    else
    dEdF0__(j) = ft_dudxy(nu, x, y, Nxyw, dNdx, dNdy, d2Ndx2, d2Ndy2) -nu*(-pi^2*y*sin(pi*x)) + beta*d2u_dx2_dy2_(j)  - ft_F(x,y);
    end
    
    
    X = w_*[x;y;b];
    dF0dw(:,1) = -nu*((2*(y^2-y)+2*(x^2-x))*x*v1.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*(v1.*grad_sigmoid(X)+x*w_(:,1).*v1.*grad2_sigmoid(X)) + 2*(x^2-x)*(2*y-1)*x*w_(:,2).*v1.*grad2_sigmoid(X) + (x^2-x)*(y^2-y)*(2*w_(:,1).*v1.*grad2_sigmoid(X)+x*w_(:,1).^2.*v1.*grad3_sigmoid(X)) + (x^2-x)*(y^2-y)*(x*w_(:,2).^2.*v1.*grad3_sigmoid(X)));
    dF0dw(:,2) = -nu*((2*(y^2-y)+2*(x^2-x))*y*v1.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*y*w_(:,1).*v1.*grad2_sigmoid(X) + 2*(x^2-x)*(2*y-1)*(v1.*grad_sigmoid(X)+y*w_(:,2).*v1.*grad2_sigmoid(X)) + (x^2-x)*(y^2-y)*y*w_(:,1).^2.*v1.*grad3_sigmoid(X) + (x^2-x)*(y^2-y)*(2*w_(:,2).*v1.*grad2_sigmoid(X)+y*w_(:,2).^2.*v1.*grad3_sigmoid(X)));
    dF0dw(:,3) = -nu*((2*(y^2-y)+2*(x^2-x))*b*v1.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*(b*w_(:,1).*v1.*grad2_sigmoid(X)) + 2*(x^2-x)*(2*y-1)*b*w_(:,2).*v1.*grad2_sigmoid(X) + (x^2-x)*(y^2-y)*(b*w_(:,1).^2.*v1.*grad3_sigmoid(X)) + (x^2-x)*(y^2-y)*(b*w_(:,2).^2.*v1.*grad3_sigmoid(X)));
    dEdw_ = dEdw_ + dEdF0__(j)*dF0dw;
   end 
   f1_w = 1/2*sum(dEdF0__.^2);
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
   v_ = v1 + delta_v;
dEdv_ = zeros(H,1);
   for j=1:N^2
    x = x_(j);
    y = y_(j);
    Nxyw = ft_Nxyw(x,y,w1,v_);
    dNdx = ft_dNdx(x,y,w1,v_);
    dNdy = ft_dNdy(x,y,w1,v_);
    d2Ndx2 = ft_d2Ndx2(x,y,w1,v_);
    d2Ndy2 = ft_d2Ndy2(x,y,w1,v_);
    
        
    if mod(iter,2) == 1 
    dEdF0__(j) = ft_dudxy(nu, x, y, Nxyw, dNdx, dNdy, d2Ndx2, d2Ndy2) -nu*(-pi^2*y*sin(pi*x))   - ft_F(x,y);
    else
    dEdF0__(j) = ft_dudxy(nu, x, y, Nxyw, dNdx, dNdy, d2Ndx2, d2Ndy2) -nu*(-pi^2*y*sin(pi*x)) + beta*d2u_dx2_dy2_(j)  - ft_F(x,y);
    end
    
    X = w1*[x;y;b];
    dF0dv = -nu*((2*(y^2-y)+2*(x^2-x))*sigmoid(X) + 2*(2*x-1)*(y^2-y)*w1(:,1).*grad_sigmoid(X) + 2*(x^2-x)*(2*y-1)*w1(:,2).*grad_sigmoid(X) + (x^2-x)*(y^2-y)*(w1(:,1).^2.*grad2_sigmoid(X)+w1(:,2).^2.*grad2_sigmoid(X)));
    dEdv_ = dEdv_ + dEdF0__(j)*dF0dv;
   end 
   f1_v = 1/2*sum(dEdF0__.^2);
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
       w1 = w1 + reshape(delta_w,H,3);
       delta_v = - alphav*Bv_inv*dEdv_old(:);
       v1 = v1 + delta_v;
dEdw = zeros(H,3);
dEdv = zeros(H,1);

beta = 10^(-8);

   for j=1:N^2
    x = x_(j);
    y = y_(j);
    
    Nxyw = ft_Nxyw(x,y,w1,v1);
    dNdx = ft_dNdx(x,y,w1,v1);
    dNdy = ft_dNdy(x,y,w1,v1);
    d2Ndx2 = ft_d2Ndx2(x,y,w1,v1);
    d2Ndy2 = ft_d2Ndy2(x,y,w1,v1);
    
    if mod(iter,2) == 1 
    dEdF0(j) = ft_dudxy(nu, x, y, Nxyw, dNdx, dNdy, d2Ndx2, d2Ndy2) -nu*(-pi^2*y*sin(pi*x))   - ft_F(x,y);
    else
    dEdF0(j) = ft_dudxy(nu, x, y, Nxyw, dNdx, dNdy, d2Ndx2, d2Ndy2) -nu*(-pi^2*y*sin(pi*x)) + beta*d2u_dx2_dy2_(j)  - ft_F(x,y);
    end    
    
    
    X = w1*[x;y;b];
    dF0dw(:,1) = -nu*((2*(y^2-y)+2*(x^2-x))*x*v1.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*(v1.*grad_sigmoid(X)+x*w1(:,1).*v1.*grad2_sigmoid(X)) + 2*(x^2-x)*(2*y-1)*x*w1(:,2).*v1.*grad2_sigmoid(X) + (x^2-x)*(y^2-y)*(2*w1(:,1).*v1.*grad2_sigmoid(X)+x*w1(:,1).^2.*v1.*grad3_sigmoid(X)) + (x^2-x)*(y^2-y)*(x*w1(:,2).^2.*v1.*grad3_sigmoid(X)));
    dF0dw(:,2) = -nu*((2*(y^2-y)+2*(x^2-x))*y*v1.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*y*w1(:,1).*v1.*grad2_sigmoid(X) + 2*(x^2-x)*(2*y-1)*(v1.*grad_sigmoid(X)+y*w1(:,2).*v1.*grad2_sigmoid(X)) + (x^2-x)*(y^2-y)*y*w1(:,1).^2.*v1.*grad3_sigmoid(X) + (x^2-x)*(y^2-y)*(2*w1(:,2).*v1.*grad2_sigmoid(X)+y*w1(:,2).^2.*v1.*grad3_sigmoid(X)));
    dF0dw(:,3) = -nu*((2*(y^2-y)+2*(x^2-x))*b*v1.*grad_sigmoid(X) + 2*(2*x-1)*(y^2-y)*(b*w1(:,1).*v1.*grad2_sigmoid(X)) + 2*(x^2-x)*(2*y-1)*b*w1(:,2).*v1.*grad2_sigmoid(X) + (x^2-x)*(y^2-y)*(b*w1(:,1).^2.*v1.*grad3_sigmoid(X)) + (x^2-x)*(y^2-y)*(b*w1(:,2).^2.*v1.*grad3_sigmoid(X)));
    dEdw = dEdw + dEdF0(j)*dF0dw;

    dF0dv = -nu*((2*(y^2-y)+2*(x^2-x))*sigmoid(X) + 2*(2*x-1)*(y^2-y)*w1(:,1).*grad_sigmoid(X) + 2*(x^2-x)*(2*y-1)*w1(:,2).*grad_sigmoid(X) + (x^2-x)*(y^2-y)*(w1(:,1).^2.*grad2_sigmoid(X)+w1(:,2).^2.*grad2_sigmoid(X)));
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
end
   iter_ = iter_ + fi_num    
   co_num, fi_num

    convergence_rate = f0;
   fprintf('convergence rate =%12.7g \n',f0);

   for j=1:N^2
    x = x_(j);
    y = y_(j);
    uh1(j) = x*(x-1)*y*(y-1)*sum(ft_Nxyw(x,y,w1,v1)) + y*sin(pi*x);
   end 


  for j=1:N^2
    uh(j) = uh1(j); 
  end  
    iter = iter + 1;
    
        eh = ut - uh;
        l2_err = sqrt( eh'*M*eh );  h1_err = sqrt( eh'*A*eh );
    l2_err__ = [l2_err__ l2_err l2_err l2_err ];
    h1_err__ = [h1_err__ h1_err h1_err h1_err ];
    
    if mod(iter,1) == 0
        ut = ft_u(xx(:),yy(:));
        subplot(161); surf(xx,yy,reshape(ut,N,N));
        subplot(162); surf(xx,yy,reshape(uh,N,N));
        subplot(163); surf(xx,yy,abs(reshape(uh,N,N)-reshape(ut,N,N)));
        axis([0 1 0 1])
        colorbar;
        subplot(164); plot([1:length(f0__)],log(f0__),'r-');
        subplot(165); plot([1:length(f0__)],log10(l2_err__),'r-');
        subplot(166); plot([1:length(f0__)],log10(h1_err__),'r-');
        getframe; % hold on;
    end
save('v1','v1');
save('w1','w1');
save('v2','v2');
save('w2','w2');
       if iter_ > 25000 
        ut = ft_u(xx(:),yy(:));
       
        eh = ut - uh;
        [M,S] = p1fem(N,1);
        l2_err = sqrt( eh'*M*eh );  h1_err = sqrt( eh'*(M+S)*eh );
        fprintf('Nodes=%4.0f  l2-error=%12.4e H1-error=%12.4e \n',N,l2_err,h1_err);
        fprintf('============================================================\n')
        break; 
       end    

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
  
function dudxy = ft_dudxy(nu, x, y, Nxyw, dNdx, dNdy, d2Ndx2, d2Ndy2)
  dudxy = -nu*sum((2*(y^2-y)+2*(x^2-x))*Nxyw + 2*(2*x-1)*(y^2-y)*dNdx + 2*(x^2-x)*(2*y-1)*dNdy + (x^2-x)*(y^2-y)*(d2Ndx2+d2Ndy2));% -nu*(-pi^2*y*sin(pi*x));

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







