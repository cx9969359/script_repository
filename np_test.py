import numpy as np

gt_bbox = np.array([[3,3,7,8],[12,10,15,12],[9,8,15,15]])
mc_bbox = np.array([[2,2,4,4,.9],[0,0,2,2,.8],[8,7,10,9,.4],[6,6,8,8,.5]]).reshape((-1,1,5))[:,:,:4]

print(gt_bbox.shape)
print(mc_bbox.shape)

xmin = np.maximum(gt_bbox[:,0], mc_bbox[:,:,0])
ymin = np.maximum(gt_bbox[:,1], mc_bbox[:,:,1])
xmax = np.minimum(gt_bbox[:,2], mc_bbox[:,:,2])
ymax = np.minimum(gt_bbox[:,3], mc_bbox[:,:,3])
w = np.maximum(xmax - xmin, 0.)
h = np.maximum(ymax - ymin, 0.)

inter = w*h


print(inter)
inter_check = np.where(inter>0,1,0)
print(inter_check)
print(inter_check.shape)

tp_array_1 = np.sum(inter_check, axis=1)
print(tp_array_1)
tp_array_1 = np.where(tp_array_1>0, 1, 0)
tp_array_1 = np.sum(tp_array_1)
print(tp_array_1)

# 筛选tp
tp_array_2 = np.sum(inter_check, axis=0)
print(tp_array_2)
tp_array_2 = np.where(tp_array_2>0, 1, 0)
tp_array_2 = np.sum(tp_array_2)
print(tp_array_2)
