function [a,b,c] = getWalnutBounds(wal_num)
a=1:501;b=a;c=a;

if wal_num == 101
    a = 45:474;
    b = 70:430;
    c = 50:475;
end

if wal_num == 103
    c = 100:420;
    b = 95:412;
    a = 60:460;
end

if wal_num == 102
    c = 95:425;
    b = 80:400;
    a = 60:445;
end

if wal_num == 104
    c = 55:420;
    b = 80:450;
    a = 40:480;
end

if wal_num == 105
    c = 45:405;
    b = 75:425;
    a = 40:460;
end

if wal_num == 106
    c = 40:430;
    b = 55:455;
    a = 30:470;
end

end
    