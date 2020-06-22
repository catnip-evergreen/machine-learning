%% code to randomise the images in matlab
data = readtable('train_speed.csv');% read data from a csv file
c = table2cell(data) ;
n1 = length(c) ;
for i =1:1:n1
    c(i,1) = strcat('/floyd/input/train_data_images_final/',c(i,1));
end    
d = [c(1,1:2),c(2,1:2)];
for i =1:1:n1-1

    d = [d;c(i,1:2),c(i+1,1:2)];
end

n = length(d) ;
d = d(2:n,:);

random_number = randperm(n-1) ;
random_number(randperm(length(random_number)));
train_data = [c(1,1:2),c(2,1:2)];
valid_data = [c(1,1:2),c(2,1:2)];
test_data =  [c(1,1:2),c(2,1:2)];
for i = 1:1:n-1
   
    idx1 = random_number(i) ;
    r1 = d(idx1,:) ;
    
    no=11;   % range desired
    p=randperm(no);
    for j=1:no
    a = p(j)-1;
    end
    if (a>=0) && (a<=1)
    valid_data = [valid_data;r1] ;
    end
    if (a==2)
    test_data = [test_data;r1] ;
    end
    if (a>2) 
    train_data = [train_data;r1];
    end
    
end    
vd = length(valid_data) ;
ted = length(test_data) ;
trd = length(train_data) ;
valid_data = valid_data(2:vd,:);
test_data = test_data(2:ted,:);
train_data = train_data(2:trd,:);
x = cell2table(valid_data);
y = cell2table(test_data);
z = cell2table(train_data) ;
writetable(x,'valid_data.csv');
writetable(y,'test_data.csv');
writetable(z,'train_data.csv');

