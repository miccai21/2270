from segmentation2d_main import parser2D, SegmentationNet as SegmentationNet2D
from main import parser3D, SegmentationNet as SegmentationNet3D

from torch.backends import cudnn


pgsegm_root = '/nas/softechict-nas-2/fresearchers/prostata/'


def slice_2D():
    opt = parser2D()
    print(opt)
    n = SegmentationNet2D(num_epochs=opt.epochs, lossname=opt.loss, optimizer=opt.optimizer, l_r=opt.learning_rate, size=opt.imageSize,
                        thresh=opt.thresh, n_workers=opt.workers, batch_size=opt.batch_size, net_name=opt.network,
                        ckpt_name=opt.ckpt_name, num_classes=opt.classes, augm_config=opt.augm_config, opt=opt,
                        job_id=opt.job_id)

    n.load()
    n.plot_contours()


def slice_3D():
    opt = parser3D()
    print(opt)
    n = SegmentationNet3D(num_epochs=opt.epochs, lossname=opt.loss, optimizer=opt.optimizer, l_r=opt.learning_rate,
                        interp_size=opt.interp_size, crop_size=opt.crop_size,
                        thresh=opt.thresh, n_workers=opt.workers, batch_size=opt.batch_size, net_name=opt.network,
                        ckpt_name=opt.ckpt_name, num_classes=opt.classes, augm_config=opt.augm_config, opt=opt,
                        job_id=opt.job_id)
    n.load()
    n.eval(d_loader=n.test_data_loader, plot_flag=True)


if __name__ == '__main__':
    cudnn.benchmark = True
    MAX_RES = 6
    # slice_2D()
    slice_3D()





