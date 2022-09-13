function [v,u]=Izhikevich_simulator(a,b,c,d,dt,T,I,t,Title,x_label,legend1,lengend2,plot1) 
v0=(25*b)/2 - (5*sqrt(5)*sqrt(5*b^2 - 50*b + 13))/2 - 125/2;
v=v0*ones(1,T);
u=v0*b*ones(1,T); 
i=2;
while i<=T
    if v(i-1)>=30 
        u(i)=u(i-1)+d;
        v(i)=c;
    else
        v(i)=v(i-1)+dt*(0.04*v(i-1)^2+5*v(i-1)+140-u(i-1)+I(i-1));
        u(i)=u(i-1)+dt*(a*(b*v(i-1)-u(i-1)));
    end
    i=i+1;
end
if plot1 == "on"
    figure();
    plot(t,v,"b")
    hold on
    plot(t,I,"r");
    grid on;
    title(Title);xlabel(x_label);legend(legend1,lengend2);
end

