% Create dot maps from head annotation
% NOTE: Only used for testing.


function d_map = create_dotmaps(gt, img_h, img_w)

d_map = zeros(img_h, img_w);
gt = gt(gt(:, 1) < img_w, :);
gt = gt(gt(:, 2) < img_h, :);

for i = 1: size(gt, 1)

x = max(1, floor(gt(i, 1)));
y = max(1, floor(gt(i, 2)));
d_map(y, x) = 1.0;

end

end


