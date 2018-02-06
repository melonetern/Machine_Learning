function sumx = mysum(x)
%获取元素个数
num = size(x);
%初始化总和
sumx = 0;
for i = 1:num(2)
sumx = sumx + x(i);
end