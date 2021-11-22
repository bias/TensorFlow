%  [y_min, x_min, y_max, x_max]


im_width = 1920;
im_height = 1080;

boxes = [
    [0.76267564 0.15663382 0.98182213 0.25091347],
 [0.75278544 0.42465407 0.9509945  0.5199447 ],
 [0.86247456 0.15181693 0.89015675 0.18354782],
 [0.8622598  0.15225783 0.8852921  0.17196146]
 ];

% invert y dim (and hence y min,max)
boxes(:,1) = im_height - boxes(:,1) * im_height;
boxes(:,2) = boxes(:,2) * im_width;
boxes(:,3) = im_height - boxes(:,3) * im_height;
boxes(:,4) = boxes(:,4) * im_width;
 
labels = [1, 2, 4, 3];
scores = [0.9998847  0.9976926  0.98078007 0.45702973];

dot_size = 33;
dot_size2 = 50;

figure(1)

hold on


p1_xmean = (boxes(1,4)+boxes(1,2))/2;
p1_ymean = (boxes(1,3)+boxes(1,1))/2;

p1_xlen = boxes(1,4) - boxes(1,2);
p1_ylen = boxes(1,1) - boxes(1,3);
p1_ratio = p1_xlen / p1_ylen;

p2_xmean = (boxes(2,4)+boxes(2,2))/2;
p2_ymean = (boxes(2,3)+boxes(2,1))/2;

p2_xlen = boxes(2,4) - boxes(2,2);
p2_ylen = boxes(2,1) - boxes(2,3);
p2_ratio = p2_xlen / p2_ylen;

grid on
pbaspect([1 1 1])

color = 'g';
scatter(boxes(1,2), boxes(1,1), dot_size, color, 'filled')
scatter(boxes(1,2), boxes(1,3), dot_size, color, 'filled')
scatter(boxes(1,4), boxes(1,1), dot_size, color, 'filled')
scatter(boxes(1,4), boxes(1,3), dot_size, color, 'filled')

scatter(p1_xmean, p1_ymean, 80, color, 'filled')

color = 'b';
scatter(boxes(2,2), boxes(2,1), dot_size, color, 'filled')
scatter(boxes(2,2), boxes(2,3), dot_size, color, 'filled')
scatter(boxes(2,4), boxes(2,1), dot_size, color, 'filled')
scatter(boxes(2,4), boxes(2,3), dot_size, color, 'filled')

scatter(p2_xmean, p2_ymean, 80, color, 'filled')

color = 'r';
scatter(boxes(3,2), boxes(3,1), dot_size2, color, 'x')
scatter(boxes(3,2), boxes(3,3), dot_size2, color, 'x')
scatter(boxes(3,4), boxes(3,1), dot_size2, color, 'x')
scatter(boxes(3,4), boxes(3,3), dot_size2, color, 'x')

color = 'k';
scatter(boxes(3,2), boxes(3,1), dot_size, color, '*')
scatter(boxes(3,2), boxes(3,3), dot_size, color, '*')
scatter(boxes(3,4), boxes(3,1), dot_size, color, '*')
scatter(boxes(3,4), boxes(3,3), dot_size, color, '*')

color = 'c';
scatter(0, 0, dot_size, color, 'filled')
scatter(0, im_height, dot_size, color, 'filled')
scatter(im_width, 0, dot_size, color, 'filled')
scatter(im_width, im_height, dot_size, color, 'filled')

axis equal

hold off