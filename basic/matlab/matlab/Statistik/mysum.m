function sumx = mysum(x)
%��ȡԪ�ظ���
num = size(x);
%��ʼ���ܺ�
sumx = 0;
for i = 1:num(2)
sumx = sumx + x(i);
end