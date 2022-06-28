import torch
import pdb, os, argparse
import imageio
from DualFastnet_res import DualFastnet
from dataset.data_RGB import get_loader
from skimage import img_as_ubyte
from metric.metric import CalFM,CalMAE,CalSM
# from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--numworkers', type=int, default=8, help='the number of workers')
opt = parser.parse_args()

dataset_path = 'xxxxx'
model = DualFastnet()

#ck path
load_name = 'xxxxx/best_epoch.pth'

split_name = load_name.split('/')[6]
print(split_name)
model.cuda()
a = torch.load(load_name)
# model = torch.nn.DataParallel(model)
model.load_state_dict(a)
model.eval()
test_datasets=['/DUT-RGBD/test_data','/LFSD','/NLPR/test_data','/RGBD135','/SSD','/STEREO','/SIP','/NJU2K','/STEREO1000']


conter = 0
F_dic ={}
Fmax_dic ={}
MAE_dic ={}
S_dic ={}
for dataset in test_datasets:
    #
    print(test_datasets[conter])

    save_path = './results-final/'+split_name+'/'+test_datasets[conter].split('/')[1]+'/'
    conter +=1
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/images/'
    gt_root = dataset_path + dataset + '/gts/'
    depth_root = dataset_path + dataset + '/depths/'
    test_loader, test_samples = get_loader(image_root, depth_root, gt_root, batchsize=1,
                                           numworkers=opt.numworkers, trainsize=opt.testsize)
    cal_fm = CalFM(num=test_samples)# cal是一个对象
    cal_mae = CalMAE(num=test_samples)
    cal_sm = CalSM(num=test_samples)
    for step, packs in enumerate(test_loader):
        input,depth, target,name = packs
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        target =torch.squeeze(target)
        depth = depth.cuda(non_blocking=True)
        n, c, h, w = depth.size()
        depth = depth.view(n, h, w, 1).repeat(1, 1, 1, 3)
        depth = depth.transpose(3, 1)
        depth = depth.transpose(3, 2)
        with torch.no_grad():
            out1u = model(input.cuda(),depth.cuda())
            output_rgb = torch.squeeze(out1u)
        predict_rgb = output_rgb.sigmoid().cpu().detach().numpy()
        max_pred_array = predict_rgb.max()
        min_pred_array = predict_rgb.min()
        if max_pred_array == min_pred_array:
            predict_rgb = predict_rgb / 255
        else:
            predict_rgb = (predict_rgb - min_pred_array) / (max_pred_array - min_pred_array)

        cal_fm.update(predict_rgb,target.data.cpu().detach().numpy())
        cal_mae.update(predict_rgb,target.data.cpu().detach().numpy())
        cal_sm.update(predict_rgb,target.data.cpu().detach().numpy())

        # 这个负责写图
        imageio.imwrite(save_path + name[0], img_as_ubyte(predict_rgb))

    _,maxf,mmf,_,_=cal_fm.show()
    mae = cal_mae.show()
    sm = cal_sm.show()
    F_dic[test_datasets[conter-1].split('/')[1]] = mmf
    Fmax_dic[test_datasets[conter-1].split('/')[1]] = maxf
    MAE_dic[test_datasets[conter-1].split('/')[1]] = mae
    S_dic[test_datasets[conter-1].split('/')[1]] =sm
print(split_name)
print("maxF-measure")
print(Fmax_dic)
print("F-measure")
print(F_dic)
print("MAE")
print(MAE_dic)
print("S-measure")
print(S_dic)