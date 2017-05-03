from __future__ import print_function
from builtins import str
import os
import numpy as np
import selectivesearch
from easydict import  EasyDict
from base_directory.utils import nms as nmsPython
from builtins import range
import cv2, copy, textwrap
from PIL import Image, ImageFont, ImageDraw
from PIL.ExifTags import TAGS

available_font = 'arial.ttf'
try:
    dummy = ImageFont.truetype(available_font, 16)
except Exception as Argument:
    print('Exception occurred while finding font {0}'.format(Argument))
    available_font = 'FreeMono.ttf'

#TODO: Region of Interest
def func_get_selective_search_RoIs(img, ss_scale, ss_sigma, ss_min_size, max_dim):
    """ 
    Selective Search
        Parameters
        ----------
            img : ndarray
                Input image
            ss_scale : int
                Free parameter. Higher means larger clusters in felzenszwalb segmentation.
            ss_sigma : float
                Width of Gaussian kernel for felzenszwalb segmentation.
            ss_min_size : int
                Minimum component size for felzenszwalb segmentation.
            max_dim :   float
                Max dimension
        Returns
        -------
            img : ndarray
                image with region label
                region label is stored in the 4th value of each pixel [r,g,b,(region)]
            regions : array of dict
                [
                    {
                        'rect': (left, top, right, bottom),
                        'labels': [...]
                    },
                    ...
                ]
    inter_area seems to give much better results esp when upscaling image 
    """
    img, scale = func_img_resize_max_dim(img, max_dim, bo_upscale=True, interpolation=cv2.INTER_AREA)
    _, ss_RoIs = selectivesearch.selective_search(img, scale=ss_scale, sigma= ss_sigma, min_size= ss_min_size)
    rects = []
    for ss_RoI in ss_RoIs:
        x,y,w,h = ss_RoI['rect']
        rects.append([x,y,x+w,y+h])
    return rects, img, scale


def func_get_grid_of_RoIs(imgWidth, imgHeight, nrGridScales, aspectRatios = [1.0]):
    rects = []
    # start adding large ROIs and then smaller ones
    for iter in range(nrGridScales):
        cellWidth = 1.0 * min(imgHeight, imgWidth) / (2 ** iter)
        step = cellWidth / 2.0

        for aspectRatio in aspectRatios:
            wStart = 0
            while wStart < imgWidth:
                hStart = 0
                while hStart < imgHeight:
                    if aspectRatio < 1:
                        wEnd = wStart + cellWidth
                        hEnd = hStart + cellWidth / aspectRatio
                    else:
                        wEnd = wStart + cellWidth * aspectRatio
                        hEnd = hStart + cellWidth
                    if wEnd < imgWidth-1 and hEnd < imgHeight-1:
                        rects.append([wStart, hStart, wEnd, hEnd])
                    hStart += step
                wStart += step
    return rects


def func_filter_RoIs(rects, maxWidth, maxHeight, roi_minNrPixels, roi_maxNrPixels,
               roi_minDim, roi_maxDim, roi_maxAspectRatio):
    filteredRects = []
    filteredRectsSet = set()
    for rect in rects:
        if tuple(rect) in filteredRectsSet: # excluding rectangles with same co-ordinates
            continue

        x, y, x2, y2 = rect
        w = x2 - x
        h = y2 - y
        assert(w>=0 and h>=0)

        # apply filters
        if h == 0 or w == 0 or \
           x2 > maxWidth or y2 > maxHeight or \
           w < roi_minDim or h < roi_minDim or \
           w > roi_maxDim or h > roi_maxDim or \
           w * h < roi_minNrPixels or w * h > roi_maxNrPixels or \
           w / h > roi_maxAspectRatio or h / w > roi_maxAspectRatio:
               continue
        filteredRects.append(rect)
        filteredRectsSet.add(tuple(rect))

    # could combine rectangles using non-maxima surpression or with similar co-ordinates
    # groupedRectangles, weights = cv2.groupRectangles(np.asanyarray(rectsInput, np.float).tolist(), 1, 0.3)
    # groupedRectangles = nms_python(np.asarray(rectsInput, np.float), 0.5)
    assert(len(filteredRects) > 0)
    return filteredRects


def func_read_RoIs(roiDir, subdir, imgFilename):
    # remove image file extension and make roi.txt file
    roi_path = os.path.join(roiDir, subdir, imgFilename[:-4] + ".roi.txt")
    rois = np.loadtxt(roi_path, np.int)
    if len(rois) == 4 and type(rois[0]) == np.int32:  # if only a single ROI in an image
        rois = [rois]
    return rois

#TODO: Generate and parse CNTK files
def func_read_Gt_annotation(img_path):
    bboxes_path = img_path[:-4] + '.bboxes.tsv'
    lbl_path = img_path[:-4] + '.bboxes.label.tsv'
    bboxes = np.array(func_read_table(bboxes_path), np.int32)
    labels = func_read_file(lbl_path)
    assert (len(bboxes) == len(labels))
    return bboxes, labels

def func_get_cntk_input_paths(cntk_file_dir, image_set):
    cntk_imgs_list_path = os.path.join(cntk_file_dir, image_set + '.txt')
    cntk_RoI_coords_path = os.path.join(cntk_file_dir, image_set + '.rois.txt')
    cntk_RoI_label_path = os.path.join(cntk_file_dir, image_set + '.roilabels.txt')
    cntk_Nr_RoI_path = os.path.join(cntk_file_dir, image_set + '.nrRois.txt')
    return  cntk_imgs_list_path, cntk_RoI_coords_path, cntk_RoI_label_path, cntk_Nr_RoI_path

def func_RoI_transform_pad_scale_param(img_width, img_height, pad_width, pad_height, bo_resize_image = True):
    scale = 1.0
    if bo_resize_image:
        assert pad_width == pad_height, 'only supported square width = height'
        scale = 1.0 * pad_width / max(img_width, img_height)
        img_width = round(img_width * scale)
        img_height = round(img_height * scale)

    target_w = pad_width
    target_h = pad_height
    w_offset = ((target_w - img_width) / 2.)
    h_offset = ((target_h - img_height) / 2.)
    if bo_resize_image and w_offset > 0 and h_offset > 0:
        raise Exception("ERROR: both offsets are > 0:", img_width, img_height, w_offset, h_offset)
    if (w_offset < 0 or h_offset < 0):
        print("ERROR: at least one offset is < 0:", img_width, img_height, w_offset, h_offset, scale)
    return target_w, target_h, w_offset, h_offset, scale

#TODO: Some Primary functions
def func_make_dir(directory_path):
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path, exist_ok=True)
        except OSError:
            if not os.path.isdir(directory_path):
                raise

def func_get_files_in_dir(directory, post_fix = ''):
    file_names = [s for s in os.listdir(directory) if not os.path.isdir(os.path.join(directory, s))]
    if not post_fix or post_fix == '':
        return file_names
    else:
        return [s for s in file_names if s.lower().endswith(post_fix)]

def func_read_file(input_file):
    #read in binary mode, to avoid problems with end-of-text characters
    with open(input_file, 'rb') as file:
        lines = file.readline()
    return [func_remove_line_end_char(s) for s in lines]

def func_read_table(input_file, delimiter = '\t', columns_keep = None):
    lines = func_read_file(input_file)
    if columns_keep is not None:
        header = lines[0].split(delimiter)
        print('deactivated function listFindItems')
        input()
        # columns_to_keep_indices = listFindItems(header, columns_keep)
        columns_to_keep_indices = []
    else:
        columns_to_keep_indices = None
    return func_split_strings(lines, delimiter, columns_to_keep_indices)

def func_get_column(table, column_index):
    column = []
    for row in table:
        column.append(row[column_index])
    return column

def func_delete_file(file_path):
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError as argument:
            print('Exception occurred func_delete_file {0}'.format(argument))
            input()

def func_delete_all_files_in_dir(directory, file_type, bo_prompt_user = False):
    if bo_prompt_user:
        #Comfirm delete all file
        user_input = input('--> INPUT: Press "y" to delete files in directory ' + directory + ": ")
        if not (user_input.lower() == 'y' or user_input.lower() == 'yes'):
            print('User input is {0}: exiting now'.format(user_input))
            exit()
        #otherwise delete all
        for filename in func_get_files_in_dir(directory):
            if file_type is None or filename.lower().endswith(file_type):
                func_delete_file(directory + '/' + filename)

def func_write_to_file(output_file, lines):
    with open(output_file, 'w') as file:
        for line in lines:
            file.write('%s\n' % line)

def func_remove_line_end_char(line):
    if line.endswith(b'\r\n'):
        return line[:-2]
    elif line.endswith(b'\n'):
        return line[:-1]
    else:
        return line


def func_split_individual_string(string, delimiter = '\t', column_to_keep_indices = None):
    if string is None:
        return None
    items = string.decode('utf-8').split(delimiter)
    if column_to_keep_indices is not None:
        print('deactivated getColumns func')
        input()
        # items = getColumns([items], columnsToKeepIndices)
        items = items[0]
    return items

def func_split_strings(strings, delimiter, column_to_keep_indices = None):
    table = [func_split_individual_string(string, delimiter, column_to_keep_indices) for string in strings]
    return table

def func_find(list_1d, func):
    return [index for (index, item) in enumerate(list_1d) if func(item)]

def func_table_to_list_1d(table, delimiter = '\t'):
    output = [delimiter.join([str(s) for s in row]) for row in table]
    return output

def func_sort_dict(dictionary, sort_index = 0, reverse_sort = False):
    return sorted(dictionary.items(), key= lambda x : x[sort_index], reverse= reverse_sort)

def func_imread(img_path, bo_throw_error_if_exif = True):
    if not os.path.exists(img_path):
        raise  Exception('Error: func_imread image path does not exist')
    rotation = func_rotation_from_exif_tag(img_path)
    if bo_throw_error_if_exif and rotation != 0:
        raise Exception('Error: func_imread exif rotation tag set, image needs to be rotated by {0} degree'.format(rotation))
    img = cv2.imread(img_path)
    if img is None:
        raise Exception('Error: func_imread can not load image at {0}'.format(img_path))
    if rotation != 0:
        rows, cols, _ = img.shape
        tmp = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
        clone_img = cv2.warpAffine(img, tmp, (cols, rows))
        return clone_img
    return img

def func_rotation_from_exif_tag(img_path):
    tag_inverted = {v: k for k, v in TAGS.items()}
    orientation_ex_if_id = tag_inverted['Orientation']
    try:
        image_exif_tag = Image.open(img_path)._getexif()
    except:
        image_exif_tag = None
    # rotate the image if orientation exif tag is present
    #
    rotation = 0
    if image_exif_tag is not None and orientation_ex_if_id is not None and orientation_ex_if_id in image_exif_tag:
        orientation = image_exif_tag[orientation_ex_if_id]
        if orientation == 1 or orientation == 0:
            rotation = 0
        elif orientation == 6:
            rotation = -90
        elif orientation == 8:
            rotation = 90
        else:
            raise Exception('Error: func_rotation_from_exif_tag orientation = {0} not supported'.format(orientation))
    return rotation

def func_img_write(img, img_path):
    cv2.imwrite(img_path, img)

def func_img_resize(img, scale, interpolation = cv2.INTER_LINEAR):
    return cv2.resize(img, (0,0), fx= scale, fy = scale, interpolation=interpolation)

def func_img_resize_max_dim(img, max_dim, bo_upscale = False, interpolation = cv2.INTER_LINEAR):
    scale = 1.0* max_dim/ max(img.shape[:2])
    if scale < 1 or bo_upscale:
        img = func_img_resize(img, scale, interpolation)
    else:
        scale = 1.0
    return img, scale

def func_img_width_height(input_img):
    width, height = Image.open(input_img).size
    return width, height

def func_img_array_width_height(input_img):
    width, height = input_img.shape
    return width, height

def func_img_show(img, wait_duration = 0, max_dim = None, window_name = 'img'):
    if isinstance(img, str):
#test if 'img' is a string
        img = cv2.imread(img)
    if max_dim is not None:
        scale_val = 1.0*max_dim / max(img.shape[:2])
        if scale_val < 1:
            img = func_img_resize(img, scale_val)
    cv2.imshow(window_name, img)
    cv2.waitKey(wait_duration)

def func_to_int(list_1d):
    return [int(float(x)) for x in list_1d]

def func_draw_rectangles(img, rectangles, color = (0, 255, 0), thickness = 2):
    for rect in rectangles:
        pt1 = tuple(func_to_int(rect[0:2]))
        pt2 = tuple(func_to_int(rect[2:]))
        cv2.rectangle(img, pt1, pt2, color, thickness)

def func_draw_crossbar(img, pt):
    (x,y) = pt
    cv2.rectangle(img, (0, y), (x, y), (255, 255, 0), 1)
    cv2.rectangle(img, (x, 0), (x, y), (255, 255, 0), 1)
    cv2.rectangle(img, (img.shape[1],y), (x, y), (255, 255, 0), 1)
    cv2.rectangle(img, (x, img.shape[0]), (x, y), (255, 255, 0), 1)

def func_pt_clip(pt, max_width, max_height):
    pt = list(pt)
    pt[0] = max(pt[0], 0)
    pt[1] = max(pt[1], 0)
    pt[0] = min(pt[0], max_width)
    pt[1] = min(pt[1], max_height)
    return pt

def func_img_convert_cv_to_pil(img):
    cv2_im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_im)

def func_img_convert_pil_to_cv(pil_img):
    rgb = pil_img.convert('RGB')
    return np.array(rgb).copy()[:, :, ::-1]

def func_pil_draw_text(pil_img, pt, text, text_width=None, color = (255,255,255), background_color = None, font = None):
    # loading default value in function call so the script won't cause errors in system where
    # "arial.ttf" cannot be found
    if font is None:
        font = ImageFont.truetype("arial.ttf", 16)
    text_y = pt[1]
    #image object for drawing
    draw = ImageDraw.Draw(pil_img)
    if text_width is None:
        #wrap text
        lines = [text]
    else:
        lines = textwrap.wrap(text, width=text_width)
    for line in lines:
        #textbox with size of width and height
        width, height = font.getsize(line)
        if background_color is not None:
            draw.rectangle((pt[0], pt[1], pt[0] + width, pt[1] + height), fill=tuple(background_color[::-1]))
        draw.text(pt, line, fill = tuple(color), font = font)
        text_y += height
    return pil_img

def func_draw_text(img, pt, text, text_width, color, background_color, font):
    if font is None:
        font = ImageFont.truetype('arial.ttf', 16)
    pil_img = func_img_convert_cv_to_pil(img)
    pil_img = func_pil_draw_text(pil_img, pt, text, text_width,color, background_color, font)
    return func_img_convert_pil_to_cv(pil_img)

def func_get_colors_palette():
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]]
    for i in range(5):
        for dim in range(0, 3):
            for s in (0.25, 0.5, 0.75):
                if colors[i][dim] != 0:
                    new_color = copy.deepcopy(colors[i])
                    new_color[dim] = int(round(new_color[dim] * s))
                    colors.append(new_color)
    return colors

def func_soft_max(vect):
    exp_vec = np.exp(vect)
    if max(exp_vec) == np.inf:
        out_vec = np.zeros(len(exp_vec))
        out_vec[exp_vec == np.inf] = vect[exp_vec == np.inf]
        out_vec = out_vec / np.sum(out_vec)
    else:
        out_vec = exp_vec / np.sum(exp_vec)
    return out_vec

def func_soft_max_2d(weights):
    e = np.exp(weights)
    dist = e/ np.sum(e, axis=1)[:, np.newaxis]
    return dist

def func_get_dictionary(keys, values, bo_convert_value_to_int = True):
    dictionary = {}
    for key,value in zip(keys, values):
        if bo_convert_value_to_int:
            value = int(value)
        dictionary[key] = value
    return dictionary

class Bbox:
    MAX_VALID_DIM = 100000
    left = top = right = bottom = None

    def __init__(self, left, top, right, bottom):
        self.left   = int(round(float(left)))
        self.top    = int(round(float(top)))
        self.right  = int(round(float(right)))
        self.bottom = int(round(float(bottom)))
        self.standardize()

    def __str__(self):
        return 'Bbox object: left = {0}, top = {1}, right = {2}, bottom = {3}'.format(self.left, self.top, self.right, self.bottom)

    def __repr__(self):
        return str(self)

    def rect(self):
        return [self.left, self.top, self.right, self.bottom]

    def max(self):
        return max([self.left, self.top, self.right, self.bottom])

    def min(self):
        return min([self.left, self.top, self.right, self.bottom])

    def width(self):
        width  = self.right - self.left + 1
        assert(width>=0)
        return width

    def height(self):
        height = self.bottom - self.top + 1
        assert(height>=0)
        return height

    def surface_area(self):
        return self.width() * self.height()

    def get_overlap_Bbox(self, bbox):
        left1, top1, right1, bottom1 = self.rect()
        left2, top2, right2, bottom2 = bbox.rect()
        overlap_left = max(left1, left2)
        overlap_top = max(top1, top2)
        overlap_right = min(right1, right2)
        overlap_bottom = min(bottom1, bottom2)
        if (overlap_left>overlap_right) or (overlap_top>overlap_bottom):
            return None
        else:
            return Bbox(overlap_left, overlap_top, overlap_right, overlap_bottom)

    def standardize(self): #NOTE: every setter method should call standardize
        new_left   = min(self.left, self.right)
        new_top    = min(self.top, self.bottom)
        new_right  = max(self.left, self.right)
        new_bottom = max(self.top, self.bottom)
        self.left = new_left
        self.top = new_top
        self.right = new_right
        self.bottom = new_bottom

    def crop(self, maxWidth, maxHeight):
        new_left   = min(max(self.left,   0), maxWidth)
        new_top   = min(max(self.top,    0), maxHeight)
        new_right  = min(max(self.right,  0), maxWidth)
        new_bottom = min(max(self.bottom, 0), maxHeight)
        return Bbox(new_left, new_top, new_right, new_bottom)

    def is_valid(self):
        if self.left>=self.right or self.top>=self.bottom:
            return False
        if min(self.rect()) < -self.MAX_VALID_DIM or max(self.rect()) > self.MAX_VALID_DIM:
            return False
        return True

def func_get_enclosing_Bbox(pts):
    left = top = float('inf')
    right = bottom = float('-inf')
    for pt in pts:
        left   = min(left,   pt[0])
        top    = min(top,    pt[1])
        right  = max(right,  pt[0])
        bottom = max(bottom, pt[1])
    return Bbox(left, top, right, bottom)

def func_bboxComputeOverlapVoc(bbox1, bbox2):
    surface_rect1 = bbox1.surface_area()
    surface_rect2 = bbox2.surface_area()
    overlap_Bbox = bbox1.get_overlap_Bbox(bbox2)
    if overlap_Bbox is None:
        return 0
    else:
        surface_overlap = overlap_Bbox.surface_area()
        overlap = max(0.0, 1.0 * surface_overlap / (surface_rect1 + surface_rect2 - surface_overlap))
        assert (0.0 <= overlap <= 1.0)
        return overlap

def func_compute_average_precision(recalls, precisions, use_07_metric=False):
    """ ap = voc_ap(recalls, precisions, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        m_recalls = np.concatenate(([0.], recalls, [1.]))
        m_precisions = np.concatenate(([0.], precisions, [0.]))

        # compute the precision envelope
        for i in range(m_precisions.size - 1, 0, -1):
            m_precisions[i - 1] = np.maximum(m_precisions[i - 1], m_precisions[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(m_recalls[1:] != m_recalls[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((m_recalls[i + 1] - m_recalls[i]) * m_precisions[i + 1])
    return ap
