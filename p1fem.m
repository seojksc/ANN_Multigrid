% p1fem.m

% To find Stiffness Matrix 'S'
%          and load vector 'F'
%     for the system without any boundary conditions
%
% D. E :  - Del u + p u = f   in [0,1]x[0,1]
% B, C :              u = q   on Gamma_d
%                 du/dn = g   on Gamma_n
% e.g.  f(x,y) = Define ft_f.m
%       p(x,y) = Define ft_p.m
%       q(x,y) = Define ft_q.m    on Gamma_d
%       g(x,y) = Define ft_g.m    on Gamma_n  
% Input data
%      h =  x-grid size 
% Output data
%      S = Stiffness matrix
%      F = Load vector
% function [app,S,F,E,x,y,h] = p1fem(N,ne)

function [M,S] = p1fem(N,ne)

h = 1/(N-1); 
nT = 2*(N-1)^2;   % nbr of triangles
nN = N^2;         % nbr of nodes
S = sparse(nN,nN); M = sparse(nN,nN); 
F = sparse(nN,1);

% Call nodetri.m  for Nodes and Triangulation
[T,x,y,bnode] = triang(N,N);
  areaK = h*h/2;    % Area of each triangle on uniform mesh
  Nd = T(1,:);
  K = [x(Nd) y(Nd)];
  [elS,elM] = elstif(K,ne);
for k = 1:nT
  Nd = T(k,:);
  K = [x(Nd) y(Nd)];
  [elS,elM] = elstif(K,ne);
 for i = 1:3
   for j = 1:3
     S(Nd(i),Nd(j)) = S(Nd(i),Nd(j)) + elS(i,j); %+ft_p(x(1),y(1),ne)*elM(i,j); %if p=c
     M(Nd(i),Nd(j)) = M(Nd(i),Nd(j)) + elM(i,j); %if p=c     
   end
 end 
end
%%%%%%%%%%%%%%%%%% Applying Neumann boundary condition %%%%%%%%%%%%%%%
% g=0.5
for i=1:N
    S(:,i*N-1)=S(:,i*N-1)+S(:,i*N);
end
for i=1:N
    S(:,N*(N-2)+i)=S(:,N*(N-2)+i)+S(:,N*(N-1)+i);
end
S = (areaK/3)*S;%2*(areaK)^2*S
%for i=1:N
 %   M(:,i*N-1)=M(:,i*N-1)+M(:,i*N);
%end
M = (areaK/3)*M;%2*(areaK)^2*M
  fxy = ft_f(x,y,ne);
F = M*fxy; % int_omega(sum[f_i*phi_i]*phi_j)dxdy

return


if ne ~= 3,
    NN = 1:nN; NN(bnode) = zeros(1,length(bnode));
    Iind = nonzeros(NN);
    D = S(Iind,bnode);% For non-homogeneous Dirichlet problem,  replace D*q to RHS
                      %  where  u = q  on \Gamma
    S = S(Iind,Iind);  % Stiffness including interior nodes
    F = F(Iind); 
%    F = F - D*[zeros(1,size(x(bnode))*0.5) ones(1,size(x(bnode))*0.5)*0.5*h]';%ft_q(x(bnode),y(bnode),ne);%
    F = F - D*[zeros(1,size(x(bnode))*0.5) ones(1,size(x(bnode))*0.5)*-0.15*h]';%ft_q(x(bnode),y(bnode),ne);%
    x = x(Iind); y = y(Iind);  % Interior points
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
E = exa_u(x,y,ne);  % Exact Solution
app = S\F;
%%%% To Plot the Solution and its approximation
if ne == 3, 
   xi = 0:h:1; yi = 0:h:1;
else
   xi=h:h:1-h; yi=h:h:1-h;
end
[xx,yy]=meshgrid(xi,yi);
xn = length(xi);yn = length(yi);
e = exa_u(xx,yy,ne);  % Exact Solution
ap = [];
for i=1:xn, ap = [ap [app(yn*(i-1)+1:yn*i)]];end
figure(1)
subplot(121);mesh(xx,yy,e),title('Exact Solution');
%axis([0 1 0 1 min(app) max(app)])
subplot(122);mesh(xx,yy,ap),title('Approximation');
%axis([0 1 0 1 min(app) max(app)])
figure(2), mesh(xx,yy,e-ap) ;
fprintf(' nx = %2.0f  dim = %4.0f  hx = %5.4f   Error = %10.9f \n', ...
          N,length(app),h,max(abs(app-E)));
          
%%% triang.m
% find node and triangles on [0,1]x[0,1]
function  [T,x,y,bnode] = triang(nx,ny);

hx = 1/(nx-1);  hy = 1/(ny-1);
ne = 2*(nx-1)*(ny-1);
x = zeros(nx*ny,1); y = x;  % (x,y) cordinates 
% find the coordinates of nodes
in = 0;
for i=1:nx
    xx = (i-1)*hx;
   for j=1:ny
     yy   = (j-1)*hy;
     in   = in+1;
     x(in) = xx;%(real distance)
     y(in) = yy;
   end
end

% find the nodes for each triangles
T = zeros(ne,3);

for i = 1:nx-1
     ine = 2*(i-1)*(ny-1);
  for j=1:ny-1
     i1 = i*ny+j;
     i2 = (i-1)*ny+1+j;
     i3 = (i-1)*ny+j;
     T(ine+2*(j-1)+1,:) = [i1 i2 i3];
     T(ine+2*j,:) = [i2 i1 i1+1];
  end
end   

bnode = [1:ny ny+1:ny:(nx-2)*ny+1 2*ny:ny:(nx-1)*ny (nx-1)*ny+1:nx*ny];
bnode = sort(bnode);


%%% elstiff.m
% Int_K( p(x,y)* phi_i * phi_j ) = 
%   [ sum_{s=4}^{6} p(b^s) * phi^r_i(a^s) * phi^r_j(a^s) ] * Area(K) / 3
%  using the fact that
%      (phi^j)(b^s) = (phi^r_j)(a^s)
%  Area(K) = abs(det(J))/2
%% If mesh is uniform and p(x,y)=c is constant, 
%%   then S is independent of K, i.e. S(K) = S for any K.

function [elS,elM] = elstif(K,ne);

% Mid-points of triangle K
xc = K(:,1); yc = K(:,2);
xm(1) = (K(1,1)+K(2,1))/2;  xm(2) = (K(2,1)+K(3,1))/2; % mid-points(real distance)
xm(3) = (K(3,1)+K(1,1))/2;  ym(1) = (K(1,2)+K(2,2))/2;    
ym(2) = (K(2,2)+K(3,2))/2;  ym(3) = (K(3,2)+K(1,2))/2;     

elS = zeros(3,3); elM = zeros(3,3);  % if p(x,y)=c
a = [1 0;0 1;0 0];  % three vertices of the reference triangle
b = [1/2 1/2;0 1/2;1/2 0];  % three mid-points of the reference triangle
J = [xc(1)-xc(3) xc(2)-xc(3); yc(1)-yc(3) yc(2)-yc(3)];
JT = (inv(J))';
grad = [1 0; 0 1; -1 -1];  grad = grad';
phiv = [1/2 0 1/2;1/2 1/2 0;0 1/2 1/2];

for i=1:3
   for j=1:3
     for s=1:3   % Three-points quadruature rule
         gri = grad(:,i);  % grphi(i, b(s,1), b(s,2))';
         grj = grad(:,j);  % grphi(j, b(s,1), b(s,2))';
         phii = phiv(i,s); % phi(i, b(s,1), b(s,2));
         phij = phiv(j,s); % phi(j, b(s,1), b(s,2));
       elS(i,j) = elS(i,j) + dot(JT*gri, JT*grj);  % p(x,y) = c
       elM(i,j) = elM(i,j) + phii*phij;  % p(x,y) = c
      end
   end
end   
% elS = (JT*grad)'*(JT*grad)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Gradients of linear basis \phi_i
function Y = grphi(i,x,y);
if      i == 1, Y = [ones(size(x)), zeros(size(x))];
elseif  i == 2, Y = [zeros(size(x)), ones(size(x))];
else,           Y = [-ones(size(x)), -ones(size(x))];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% linear basis \phi_i
function Y = phi(i,x,y);
if      i == 1, Y = x;
elseif  i == 2, Y = y;
else,           Y = 1-x-y;
end


%%% data_d.m

% Define the functions  f, p, q and exact sol. u
%     which are called by  glstiff.m
%
% D. E :  - Del u + p u = f   in [0,1]x[0,1]
% B, C :              u = 0   on Gamma

% data for Homogeneous boundary Conditions
function f = ft_f(x,y,ne);
if ne == 1,
   f = 2*(x-x.*x) + 2*(y-y.*y); %+x.*(1-x).*y.*(1-y);
elseif ne == 2,
   f = 2*4*pi^2*sin(2*pi*x).*sin(2*pi*y);
else
   f = (8*pi^2+1)*cos(2*pi*x).*cos(2*pi*y);
end

% function  p(x,y)
function p = ft_p(x,y,ne);
if ne == 1,
p = zeros(size(x));
elseif ne == 2,
p = zeros(size(x));
else
p = ones(size(x)); 
end

% function  q(x,y)
function Q = ft_q(x,y,ne);
if ne == 1,
Q = zeros(size(x));
elseif ne == 2,
Q = zeros(size(x));
else
Q = zeros(size(x));
end

% Exact solution  u(x,y)
function U = exa_u(x,y,ne);
if ne == 1,
U = x.*(1-x).*y.*(1-y);
elseif ne == 2,
U = sin(2*pi*x).*sin(2*pi*y);
else
U = cos(2*pi*x).*cos(2*pi*y);
end

