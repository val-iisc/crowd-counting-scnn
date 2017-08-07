% Creates test set

function create_test_set()

image_fold = './ST_part_A/test_data/images';
gt_fold = './ST_part_A/test_data/ground_truth';

final_image_fold = './test/images';
final_gt_fold = './test/gt';

mkdir(final_image_fold);
mkdir(final_gt_fold);

jpgFiles = dir([image_fold, '/*.jpg']);
numfiles = length(jpgFiles);

parfor i = 1: numfiles
    [~, f_1] = fileparts(jpgFiles(i).name);
    f = sprintf('%s.jpg',f_1);
    I = imread(fullfile(image_fold, f));
    file_name = f_1;
    mat_filename = sprintf('GT_%s.mat', file_name);

    ann = load(fullfile(gt_fold, mat_filename));
    gt = ann.image_info{1,1}.location;
    count = 1;
    
    % create density map
    d_map_h = floor(floor(double(size(I, 1)) / 2.0) / 2.0);
    d_map_w = floor(floor(double(size(I, 2)) / 2.0) / 2.0);

    % Density Map with Geometry-Adaptive Kernels (see MCNN code: 
    % Single-Image Crowd Counting via Multi-Column Convolutional 
    % Neural Network; CVPR 2017)
    % d_map = create_density(gt / 4.0, d_map_h, d_map_w);

    % Dot maps for testing
    d_map = create_dotmaps(gt / 4.0, d_map_h, d_map_w);

    p_h = floor(double(size(I, 1)) / 3.0);
    p_w = floor(double(size(I, 2)) / 3.0);
    d_map_ph = floor(floor(p_h / 2.0) / 2.0);
    d_map_pw = floor(floor(p_w / 2.0) / 2.0);

    % create non-overlapping patches of images and density maps
    py = 1;
    py2 = 1;
    for j = 1: 3
        px = 1;
        px2 = 1;
        for k = 1: 3
            final_image = double(I(py: py + p_h - 1, px: px + p_w - 1, :));
            final_gt = d_map(py2: py2 + d_map_ph - 1, px2: px2 + d_map_pw - 1);
            px = px + p_w;
            px2 = px2 + d_map_pw;
            if size(final_image, 3) < 3
                final_image = repmat(final_image, [1, 1, 3]);
            end
            image_name = sprintf('%s_%d.jpg', file_name, count);
            gt_name = sprintf('%s_%d.mat', file_name, count);
            imwrite(uint8(final_image), fullfile(final_image_fold, image_name));
            do_save(fullfile(final_gt_fold, gt_name), final_gt);
            count=count+1;
        end
        py = py + p_h;
        py2 = py2 + d_map_ph;
    end
end

end


function do_save(gt_name, final_gt)

save(gt_name, 'final_gt');

end


