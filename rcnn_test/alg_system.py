import mxnet as mx

from symdata.bbox import im_detect
from symdata.loader import load_test, load_test_img, generate_batch
from symnet.model import load_param, check_shape


# CLASSES_NAME = ["__background__",
# 'hsil','lsil', 'ep', 'ec', 'yy','normal','agc-fn']


class alg_system:
    def __init__(self, args, ctx):
        if ctx is not None:
            ctx = mx.gpu(int(ctx))
        else:
            ctx = mx.cpu(0)

        sym = get_network(args.network, args)
        arg_params, aux_params = load_param(args.params, ctx=ctx)
        self.mod = mx.module.Module(sym, ['data', 'im_info'], None, context=ctx)

        data_shapes = [('data', (1, 3, args.img_long_side, args.img_long_side)), ('im_info', (1, 3))]
        label_shapes = None
        check_shape(sym, data_shapes, arg_params, aux_params)

        self.mod.bind(data_shapes, label_shapes, for_training=False)
        self.mod.init_params(arg_params=arg_params, aux_params=aux_params)
        self.args = args

    def infer(self, img):
        args = self.args
        im_tensor, im_info, im_orig = load_test(img, \
                                                short=args.img_short_side, \
                                                max_size=args.img_long_side, \
                                                mean=args.img_pixel_means, \
                                                std=args.img_pixel_stds)

        data_batch = generate_batch(im_tensor, im_info)

        self.mod.forward(data_batch)
        rois, scores, bbox_deltas = self.mod.get_outputs()
        rois = rois[:, 1:]
        scores = scores[0]
        bbox_deltas = bbox_deltas[0]
        im_info = im_info[0]

        # decode detection
        det = im_detect(rois, scores, bbox_deltas, im_info, \
                        bbox_stds=args.rcnn_bbox_stds, nms_thresh=args.rcnn_nms_thresh, \
                        conf_thresh=args.rcnn_conf_thresh)

        result = {}
        for n in range(1, len(args.CLASSES_NAME)):
            result[args.CLASSES_NAME[n]] = []

        for [cls, conf, x1, y1, x2, y2] in det:
            if cls > 0:
                result[args.CLASSES_NAME[int(cls)]].append([x1, y1, x2, y2, conf])

        return result

    def infer_image(self, img):
        args = self.args
        im_tensor, im_info, im_orig = load_test_img(img, \
                                                    short=args.img_short_side, \
                                                    max_size=args.img_long_side, \
                                                    mean=args.img_pixel_means, \
                                                    std=args.img_pixel_stds)

        data_batch = generate_batch(im_tensor, im_info)

        self.mod.forward(data_batch)
        rois, scores, bbox_deltas = self.mod.get_outputs()
        rois = rois[:, 1:]
        scores = scores[0]
        bbox_deltas = bbox_deltas[0]
        im_info = im_info[0]

        # decode detection
        det = im_detect(rois, scores, bbox_deltas, im_info, \
                        bbox_stds=args.rcnn_bbox_stds, nms_thresh=args.rcnn_nms_thresh, \
                        conf_thresh=args.rcnn_conf_thresh)

        result = {}
        for n in range(1, len(args.CLASSES_NAME)):
            result[args.CLASSES_NAME[n]] = []

        for [cls, conf, x1, y1, x2, y2] in det:
            if cls > 0:
                result[args.CLASSES_NAME[int(cls)]].append([x1, y1, x2, y2, conf])

        return result


def get_vgg16_test(args):
    from symnet.symbol_vgg import get_vgg_test
    if not args.params:
        args.params = 'model/vgg16-0010.params'
    args.img_pixel_means = (123.68, 116.779, 103.939)
    args.img_pixel_stds = (1.0, 1.0, 1.0)
    args.net_fixed_params = ['conv1', 'conv2']
    args.rpn_feat_stride = 16
    args.rcnn_feat_stride = 16
    args.rcnn_pooled_size = (7, 7)
    return get_vgg_test(anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                        rpn_feature_stride=args.rpn_feat_stride, rpn_pre_topk=args.rpn_pre_nms_topk,
                        rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh,
                        rpn_min_size=args.rpn_min_size,
                        num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                        rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=args.rcnn_batch_size)


def get_resnet18_test(args):
    from symnet.symbol_resnet import get_resnet_test
    if not args.params:
        args.params = 'model/resnet-18-0000.params'
    args.img_pixel_means = (0.0, 0.0, 0.0)
    args.img_pixel_stds = (1.0, 1.0, 1.0)
    args.rpn_feat_stride = 16
    args.rcnn_feat_stride = 16
    args.rcnn_pooled_size = (14, 14)
    return get_resnet_test(anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                           rpn_feature_stride=args.rpn_feat_stride, rpn_pre_topk=args.rpn_pre_nms_topk,
                           rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh,
                           rpn_min_size=args.rpn_min_size,
                           num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                           rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=args.rcnn_batch_size,
                           units=(2, 2, 2, 2), filter_list=(64, 128, 256, 512), bottle_neck=False)


def get_resnet50_test(args):
    from symnet.symbol_resnet import get_resnet_test
    if not args.params:
        args.params = 'model/resnet50-0010.params'
    args.img_pixel_means = (0.0, 0.0, 0.0)
    args.img_pixel_stds = (1.0, 1.0, 1.0)
    args.rpn_feat_stride = 16
    args.rcnn_feat_stride = 16
    args.rcnn_pooled_size = (14, 14)
    return get_resnet_test(anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                           rpn_feature_stride=args.rpn_feat_stride, rpn_pre_topk=args.rpn_pre_nms_topk,
                           rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh,
                           rpn_min_size=args.rpn_min_size,
                           num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                           rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=args.rcnn_batch_size,
                           units=(3, 4, 6, 3), filter_list=(256, 512, 1024, 2048))


def get_resnet101_test(args):
    from symnet.symbol_resnet import get_resnet_test
    if not args.params:
        args.params = 'model/resnet101-0010.params'
    args.img_pixel_means = (0.0, 0.0, 0.0)
    args.img_pixel_stds = (1.0, 1.0, 1.0)
    args.rpn_feat_stride = 16
    args.rcnn_feat_stride = 16
    args.rcnn_pooled_size = (14, 14)
    return get_resnet_test(anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                           rpn_feature_stride=args.rpn_feat_stride, rpn_pre_topk=args.rpn_pre_nms_topk,
                           rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh,
                           rpn_min_size=args.rpn_min_size,
                           num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                           rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=args.rcnn_batch_size,
                           units=(3, 4, 23, 3), filter_list=(256, 512, 1024, 2048))


def get_network(network, args):
    networks = {
        'vgg16': get_vgg16_test,
        'resnet50': get_resnet50_test,
        'resnet101': get_resnet101_test,
        'resnet18': get_resnet18_test
    }
    if network not in networks:
        raise ValueError("network {} not supported".format(network))
    return networks[network](args)
