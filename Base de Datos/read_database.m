flag = 1; %robust stabilizable systems
%flag = 2; %robust stabilizable systems but nonQuadratic stabilizable

if flag == 1
    load DB_ssf_RS_500_c;
else
    load DB_ssf_nonQS_100_c;
end

for inputs=1:2
    if inputs == 1
        order_ini = 2;
    else
        order_ini = 3;
    end
    for order=order_ini:5
        for vertices=2:5
            for i=1:cases
                A = BASE{order,inputs,vertices,i}.A;
                B = BASE{order,inputs,vertices,i}.B;
                %K = BASE{order,inputs,vertices,i}.K % a known robust stabilizing gain 
                
                %Apply your test here ...
            end
            fprintf('Finished case (n,m,N) = (%d,%d,%d)\n',order,inputs,vertices);
        end
    end
end
