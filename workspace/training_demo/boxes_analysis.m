csv_file_name = "lasha_boxes.csv";
A = readmatrix( csv_file_name );


figure(1)
hold on

grid on
pbaspect([1 1 1])
dot_size = 20;
color = 'g';

entries = size(A,1);
last_frame = -1;
for i = 1:entries
    
    rowcell = num2cell( A(i,:) );
    [frame, type, score, ymin, xmin, ymax, xmax, xmean, ymean, xlen, ylen, ratio] = deal(rowcell{:});
    if type == 1
        scatter(xmean, ymean, dot_size, color, 'filled')
    elseif type == 2
        scatter(xmean, ymean, dot_size, 'b', 'filled') 
    elseif type == 3
        scatter(xmean, ymean, dot_size, 'k', 'filled')
    elseif type == 4
        scatter(xmean, ymean, dot_size, 'r', 'filled')
    end

end

axis equal
hold off