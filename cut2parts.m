% read data
mainpath = '/Volumes/Alexssd/dataset/LSPCCB/bildstein3/snapshots/NBD_wider_patch_test';

data_path = strcat( mainpath, '/*.h5');
data_files = dir(data_path);

%mkdir /Users/aleex/CCVCL/dataset/oakland/cut;
mkdir /Volumes/Alexssd/dataset/LSPCCB/bildstein3/snapshots/NBD_wider_patch_test_cut

num = 0;
point_num = 448;


for n=1:length(data_files)
    data_path = strcat( mainpath, '/',data_files(n).name);

%     h5disp(data_path);
    data = h5read(data_path,'/data');
    label = h5read(data_path,'/label');
    
    numData = length(data(1, :));

    xyzPoints = data';
    xyzLabel = label;
     %figure;
     %pcshow(xyzPoints);
     %title('Original');
    out_path = strcat('/Volumes/Alexssd/dataset/LSPCCB/bildstein3/snapshots/NBD_wider_patch_test_cut/',data_files(n).name);
 
    num = num + 1;
    cut_show1 = 0;
    cut_show2 = 0;
    total_count = 0;
%%%%%%%%%%%%%%%%  random cut part of the object  %%%%%%%%%%%%%%%
    % create 15 ramdom plains
    count = 0;
    while count < 15
        % a * x + b * y + c * z = 0
        points = xyzPoints;
        a = (rand - 0.5) * 2;
        b = (rand - 0.5) * 2;
        c = (rand - 0.5) * 2;

        points(:, 1) = points(:, 1) * a;
        points(:, 2) = points(:, 2) * b;
        points(:, 3) = points(:, 3) * c;
        S = sum(points,2);

        A1 = S >= 0;
        cut1 = xyzPoints(A1 ~= 0,:);
        length(cut1);

        A2 = S < 0;
        cut2 = xyzPoints(A2 ~= 0,:);
        length(cut2);
        
        cut_show1 = cut1;
        cut_show2 = cut2;

        if (length(cut1(:, 1)) > point_num) && (length(cut2(:, 1)) > point_num)
            %fprintf('hit \n')
            cut1 = cut1';
            cut1_path = strcat( '/cut',num2str(count*2 + 1));
            h5create(out_path, cut1_path,[length(cut1(:, 1)) length(cut1(1,:))],'Datatype','single');
            h5write(out_path,cut1_path ,cut1);

            cut2 = cut2';
            cut2_path = strcat( '/cut',num2str(count*2 + 2));
            h5create(out_path,cut2_path,[length(cut2(:, 1)) length(cut2(1,:))],'Datatype','single');
            h5write(out_path, cut2_path,cut2);

            count = count + 1;

        end
        %h5create(out_path,'/label',[1],'Datatype','int64');
        %h5write(out_path,'/label',xyzLabel);
        %h5disp(out_path);
    end
    total_count = total_count + 1;
    if mod(total_count, 50) == 0
        figure;
        pcshow(cut_show1);
        title('cut1');
        figure;
        pcshow(cut_show2);
        title('cut2');
    end
    
    h5create(out_path,'/label',[1],'Datatype','uint8');
    h5write(out_path,'/label',xyzLabel);
    h5disp(out_path);
    num;
    processing = data_files(n).name;
end

