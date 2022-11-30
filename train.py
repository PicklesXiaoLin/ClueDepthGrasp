import time
import argparse
import datetime

import torch
import torch.nn as nn
#import torch.nn.utils as utils
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import cv2
from model import Model
from loss import ssim
from data import getTrainingTestingData
from utils import AverageMeter, DepthNorm, colorize,DepthNorm_grad, DepthNorm_plus, depth2rgb
from config import config
import os
from Myloss import Normal
#import matplotlib.pyplot as plt
import numpy as np
from Myloss import Sobel
import matplotlib.pyplot as plt
from pennet import InpaintGenerator, Discriminator
from core.loss import AdversarialLoss, PerceptualLoss, StyleLoss, VGG19
from core.utils import set_seed, set_device, Progbar, postprocess
import torch.nn.functional as F
from pennet import Discriminator, Discriminator_sobel
os.environ['CUDA_VISIBLE_DEVICE'] = "0"
torch.cuda.set_device(0)

print(torch.cuda.is_available())
Norm = Normal(shape=[config.bs, 1, 240, 320]).cuda()
sobel_ = Sobel(winsize=3).cuda()


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--epochs', default=config.TRAIN_EPOCH, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=32, type=int, help='batch size')
    args = parser.parse_args()

    # ###################################################################
    # # Create model
    model = Model().cuda()
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    netD = Discriminator(in_channels=1, use_sigmoid=True).cuda()
    torch.backends.cudnn.benchmark = True
    netD_edge = Discriminator_sobel(in_channels=1, use_sigmoid=True).cuda()

    print('Transformer Model created!!!')

    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    optimD = torch.optim.Adam(netD.parameters(), lr=1e-5, betas=(0.5, 0.999))
    optimD_edge = torch.optim.Adam(netD_edge.parameters(), lr=1e-4, betas=(0.5, 0.999))

    batch_size = args.bs
    prefix = 'densenet_' + str(batch_size)

    # Load data
    train_loader, val_loader, test_loader, real_val_loader, real_test_loader, additinal_loader= getTrainingTestingData(batch_size=batch_size)
    # train_loader, test_loader, val_loader= getTrainingTestingData(batch_size=batch_size)
    # train_loader, val_loader= getTrainingTestingData(batch_size=batch_size)

    # Logging
    writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.bs), flush_secs=30)

    # Loss
    l1_criterion = nn.L1Loss()
    adversarial_loss = set_device(AdversarialLoss(type="nsgan"))

    niter = 0
    if config.load_model and config.model_name != '':
        model.load_state_dict(torch.load('Checkpoints/%s' % config.model_name))
        netD.load_state_dict(torch.load('Checkpoints/netD-%s' % config.model_name))
        netD_edge.load_state_dict(torch.load('Checkpoints/netD_edge-%s' % config.model_name))
        niter = int(config.model_name.split('_')[-3])
        print('Loading success -> %s' % config.model_name)
        print('Loading success -> netD-%s' % config.model_name)

    loss_l1 = AverageMeter()
    loss_ssim = AverageMeter()
    loss_edge = AverageMeter()
    loss_fake_gen = AverageMeter()
    loss_fake_dis = AverageMeter()
    loss_fake_dis_edge = AverageMeter()

    if config.load_model and config.model_name != '':
        for niter in range(0,8888):
            avg_real_test_loss = 0
            avg_real_val_loss = 0
            avg_syn_test_loss = 0
            avg_syn_val_loss = 0
            print("--------------- nieter: {} ---------------".format(niter))
            start = time.time()
            n=0
            avg_syn_test_loss = LogProgress_test(model, writer, test_loader, niter, cum_num=n,
                                                 last_105=avg_syn_test_loss)
            avg_syn_val_loss = LogProgress_val(model, writer, val_loader, niter, cum_num=n, last_105=  avg_syn_val_loss)

            avg_real_val_loss = LogProgress_val_real(model, writer, real_val_loader, niter, cum_num=n, last_105= avg_real_val_loss)
            avg_real_test_loss = LogProgress_test_real(model, writer, real_test_loader, niter, cum_num=n, last_105=avg_real_test_loss)
            print("syn_val={}".format(avg_syn_val_loss))
            print("syn_test={}".format(avg_syn_test_loss))
            print("real_val={}".format(avg_real_val_loss))
            print("real_test={}".format(avg_real_test_loss))

    # Start training...
    # if False:
    for epoch in range(0, args.epochs):
        end = time.time()
        print("******************************** Epoch : {} **********************************".format(epoch))

        # Switch to train mode
        model.train()
        netD.train()
        netD_edge.train()
        ###############################################################
        netD_time = 0
        dataloader_time = 0
        generator_1_time = 0
        generator_2_time = 0
        generator_3_time = 0
        other_1_time = 0
        other_2_time = 0
        l_depth_sum = 0
        l_sobel_sum = 0
        l_ssim_sum = 0
        for i, sample_batched in enumerate(train_loader):
            # -------------------------------------------------Other_1----------------------------------------------------------------
            other_1_start = time.time()
            # 输出一个周期训练进度
            step = int( (100*batch_size) / (40454) * (i - 0 + 1))
            str1 = '\r[%3d%%] %s' % (step, '>' * step)
            print(str1, end='', flush=True)

            # 导入已有的模型
            optimizer.zero_grad()
            other_1_time += time.time()-other_1_start
            # -------------------------------------------------Other_1----------------------------------------------------------------
            # -------------------------------------------------dataloader----------------------------------------------------------------
            dataloader_start = time.time()
            #准备样本跟目标数据non_blocking=True
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
            dataloader_time += time.time()-dataloader_start
            # -------------------------------------------------dataloader----------------------------------------------------------------
            # # # 记录学习率的变化
            # add_summary(dis_writer, 'lr/dis_lr', optimD.param_groups[0]['lr'], niter)  # summary the process
            # add_summary(gen_writer, 'lr/gen_lr', optimG.param_groups[0]['lr'], niter)

            #标准化数据、取数据
            # depth_gt = DepthNorm(depth_gt)
            # colors = image[:, :3, :, :]
            # depths = image[:, 3:4, :, :]
            # normals = image[:, 4:5, :, :]
            # outlines = image[:, 5:6, :, :]
            # masks = image[:, 6:7, :, :]
            # depths_masked = (depth_gt * (1 - masks).float()) + masks
            # inputs = torch.cat((depths_masked, masks), dim=1)


            # 将原图片导入到生成器
            # input_1: inputs(depth_masked+masks) 2*256*128
            # input_2: masks 1*256*128
            # input_3: colors 3*256*128
            # input_4: normals 1*256*128
            # input_5: outlines 1*256*128
            # output_1: feats [1*8*4, 1*16*8, 1*32*16, 1*64*32, 1*128*64]
            # output_2: pre_img 1*256*144
            # feats, pred_img = netG(inputs, masks, colors, normals, outlines)
            # comp_img = (1 - masks) * depth_gt + masks * pred_img

            # 将原图片导入到判别器
            # input: depths 1*256*128
            # output: dis_fake_feat 1*27*27
            # dis_real_feat = netD(depth_gt)
            # dis_fake_feat = netD(comp_img.detach())
            # dis_real_loss = adversarial_loss(dis_real_feat, True, True)
            # dis_fake_loss = adversarial_loss(dis_fake_feat, False, True)
            # dis_loss = (dis_real_loss + dis_fake_loss) / 2
            # add_summary(dis_writer, 'Train/dis_fake_loss', dis_fake_loss.item(), niter)
            #
            # # 判别器后向传播 更新损失
            # optimD.zero_grad()
            # dis_loss.backward()
            # optimD.step()

            # 判别器生成器对抗损失
            # gen_fake_feat = netD(comp_img)  # in: [rgb(3)]
            # gen_fake_loss = adversarial_loss(gen_fake_feat, True, False)
            # gen_loss += gen_fake_loss * 0.1
            # add_summary(gen_writer, 'Train/gen_fake_loss', gen_fake_loss.item(), niter)
            #
            # #生成器点损失 洞损失
            # hole_loss = l1_loss(pred_img * masks, depth_gt * masks) / torch.mean(masks)
            # gen_loss += hole_loss * 6
            # add_summary(gen_writer, 'Train/hole_loss', hole_loss.item(), niter)
            # valid_loss = l1_loss(pred_img * (1 - masks), depth_gt * (1 - masks)) / torch.mean(1 - masks)
            # gen_loss += valid_loss * 1
            # add_summary(gen_writer, 'Train/valid_loss', valid_loss.item(), niter)
            #
            # if feats is not None:
            #     pyramid_loss = 0
            #     for _, f in enumerate(feats):
            #         pyramid_loss += l1_loss(f, F.interpolate(depth_gt, size=f.size()[2:4], mode='bilinear',align_corners=True))
            #     gen_loss += pyramid_loss * 0.5
            #     add_summary(gen_writer, 'Train/pyramid_loss', pyramid_loss.item(), niter)
            # #gen_loss_AM.update(gen_loss.data.item(), inputs.size(0))
            #
            # # 生成器后向传播 更新损失
            # optimG.zero_grad()
            # gen_loss.backward()
            # optimG.step()
            # -------------------------------------------------Generator_1----------------------------------------------------------------
            # Normalize depth
            # depth_n = DepthNorm(depth_gt)########################

            # # Predict swtitch pred_edge and edge
            output, picture = model(image)

            ones_mask = image[:, 6:7, :, :]
            output_n = output   # 9.4 comparison local * (ones_mask)
            depth_n = depth_gt * (ones_mask) * 255

            pred_edge_n = sobel_(output_n) # local edge
            edge_n = sobel_(depth_n)
            # pred_edge = sobel_(output) # global edge
            # edge = sobel_(depth_gt)

            # -------------------------------------------------Generator_1----------------------------------------------------------------

            # -------------------------------------------------netD----------------------------------------------------------------

            # Predict Dis
            # plt.subplot(211),plt.imshow(output_n.cpu().detach().numpy()[0,0,:,:])
            # plt.subplot(212),plt.imshow(depth_n.cpu().detach().numpy()[0,0,:,:])
            # plt.show()
            dis_real_feat = netD(depth_n)
            dis_real_feat_edge = netD_edge(edge_n)

            dis_fake_feat = netD(output_n.detach())
            dis_fake_feat_edge = netD_edge(pred_edge_n.detach())

            dis_real_loss = adversarial_loss(dis_real_feat, True, True)
            dis_real_loss_edge = adversarial_loss(dis_real_feat_edge, True, True)
            dis_fake_loss = adversarial_loss(dis_fake_feat, False, True)
            dis_fake_loss_edge = adversarial_loss(dis_fake_feat_edge, False, True)

            dis_loss = (dis_real_loss + dis_fake_loss) / 2
            dis_loss_edge = (dis_real_loss_edge + dis_fake_loss_edge) / 2

            optimD.zero_grad()
            optimD_edge.zero_grad()
            dis_loss.backward()
            dis_loss_edge.backward()
            optimD.step()
            optimD_edge.step()

            # Predict Gen
            gen_fake_feat = netD(output_n)  # in: [rgb(3)]
            gen_fake_feat_edge = netD_edge(pred_edge_n)  # in: [rgb(3)]
            gen_fake_loss = adversarial_loss(gen_fake_feat, True, False)
            gen_fake_loss_edge = adversarial_loss(gen_fake_feat_edge, True, False)
            # gen_adversarial_loss =  (2*gen_fake_loss_edge)
            # gen_adversarial_loss = (0.5*gen_fake_loss) + (0.2*gen_fake_loss_edge)
            # gen_adversarial_loss =  (0.5*gen_fake_loss)

            # -------------------------------------------------netD----------------------------------------------------------------

            # -------------------------------------------------Generator_2----------------------------------------------------------------
            generator_2_start = time.time()
            # Compute the loss
            l_sobel = nn.L1Loss()(edge_n, pred_edge_n) #grad only about masked edge
            l_depth = l1_criterion(output_n, depth_n) #depth_n - predict_depth  output = > output_n
            l_ssim = torch.clamp((1 - ssim(output_n, depth_n, val_range=1000.0 / 10)) * 0.5, 0, 1)


            loss =  (0.5 * l_ssim) + (0.2 * l_depth)  + (0.8* l_sobel)  # no global edge
            # loss = (0.8 * l_sobel) + (0.5 * l_ssim) + (0.2 * l_depth) # no gan


            ##Update step
            loss_l1.update(l_depth.data.item(), image.size(0))
            loss_ssim.update(l_ssim.data.item(), image.size(0))
            loss_edge.update(l_sobel.data.item(), image.size(0))
            loss_fake_gen.update(gen_fake_loss.data.item(), image.size(0))
            loss_fake_dis.update(dis_loss.data.item(), image.size(0))
            loss_fake_dis_edge.update(dis_loss_edge.data.item(), image.size(0))

            # -------------------------------------------------Generator_2----------------------------------------------------------------

            loss.backward()
            optimizer.step()

            # -------------------------------------------------Other_2----------------------------------------------------------------

            # Log progress
            niter += 1

            # -------------------------------------------------Other_2----------------------------------------------------------------

        n = 1

        # Log to tensorboard
        writer.add_scalar('Train/L1', loss_l1.val, epoch)
        writer.add_scalar('Train/SSIM', loss_ssim.val, epoch)
        writer.add_scalar('Train/EDGE', loss_edge.val, epoch)
        writer.add_scalar('Train/Gen', loss_fake_gen.val, epoch)
        writer.add_scalar('Train/Dis', loss_fake_dis.val, epoch)
        writer.add_scalar('Train/Dis_Sobel', loss_fake_dis_edge.val, epoch)

        avg_syn_val_loss = LogProgress_val(model, writer, val_loader, epoch, cum_num=n, last_105=0)
        avg_syn_test_loss = LogProgress_test(model, writer, test_loader, epoch, cum_num=n,
                                             last_105=0)
        avg_real_val_loss = LogProgress_val_real(model, writer, real_val_loader, epoch, cum_num=n,
                                                 last_105=0)
        avg_real_test_loss = LogProgress_test_real(model, writer, real_test_loader, epoch, cum_num=n,
                                                   last_105=0)
        additinal_val_loss = LogProgress_additional_val(model, writer, additinal_loader, epoch, cum_num=n,
                                                   last_105=0)
        if avg_syn_val_loss > 0.50 :
            if avg_syn_test_loss > 0.50:
                if avg_real_val_loss > 0.50:
                    if avg_real_test_loss > 0.50:
                        if not os.path.exists('Checkpoints/%s' % config.save_name):
                            os.makedirs('Checkpoints/%s' % config.save_name)
                        if not os.path.exists('Checkpoints/netD-%s' % config.save_name):
                            os.makedirs('Checkpoints/netD-%s' % config.save_name)
                        if not os.path.exists('Checkpoints/netD_edge-%s' % config.save_name):
                            os.makedirs('Checkpoints/netD_edge-%s' % config.save_name)

                        EFFECT = "-{:.4}-{:.4}-{:.4}-{:.4}".format(avg_syn_val_loss, avg_syn_test_loss,
                                                                       avg_real_val_loss, avg_real_test_loss)
                        save_name = '%s/net_params_%s_%s.pkl' % (
                            config.save_name, epoch, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
                        save_name_netD = 'netD-%s/net_params_%s_%s.pkl' % (
                            config.save_name, epoch, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
                        save_name_netD_edge = 'netD_edge-%s/net_params_%s_%s.pkl' % (
                            config.save_name, epoch, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
                        torch.save(model.state_dict(), 'Checkpoints/%s' % save_name +"%s" %EFFECT )
                        torch.save(netD.state_dict(), 'Checkpoints/%s' % save_name_netD +"%s" %EFFECT)
                        torch.save(netD_edge.state_dict(), 'Checkpoints/%s' % save_name_netD_edge +"%s" %EFFECT)
                        print('save success -> %s' % save_name +"%s" %EFFECT )
        print("")
        print("syn_val={}".format(avg_syn_val_loss),"syn_test={}".format(avg_syn_test_loss))
        print("real_val={}".format(avg_real_val_loss),"real_test={}".format(avg_real_test_loss))
        print("additinal_val={}".format(additinal_val_loss))

            # print('save success -> %s' % save_name_netD)
            # print('save success -> %s' % save_name_netD_edge)

        # print("---------------------------------- 时间消耗----------------------------------".format((time.time() - end) / 60))
        # print("Data = {}min".format((dataloader_time) / 60))
        # print("Other = {}min".format((other_1_time+other_2_time) / 60))
        # print("NetD = {}min".format((netD_time) / 60))
        # print("Generator_1 = {}min".format((generator_1_time) / 60))
        # print("Generator_2 = {}min".format((generator_2_time) / 60))
        # print("Generator_3 = {}min".format((generator_3_time) / 60))
        # print("test_time = {}min".format((time.time()-test_time) / 60))
        print("-----------------------total = {}min-----------------------".format((time.time()-end) / 60))

def LogProgress_val_1(model, writer, test_loader, global_step, cum_num=0, last_105=0):
    model.eval()

    sequential = test_loader
    sample_batched = next(iter(sequential))

    image = torch.autograd.Variable(sample_batched['image'].cuda())
    depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

    # Normalize depth
    # depth_n = DepthNorm(depth)######################

    output, picture = model(image)
    ones_mask = image[:, 6:7, :, :]
    #
    output_n = output * (ones_mask) * 255
    depth_n = depth * (ones_mask) * 255

    # maxRatio = torch.max(output_n / depth_n, depth_n / output_n)
    diff = (torch.max(output_n / depth_n, depth_n / output_n))
    diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.25, 0.6 , diff.cpu().detach().numpy() ))
    diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.10, 0.4 , diff.cpu().detach().numpy() ))
    diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.05, 0.2 , diff.cpu().detach().numpy() ))
    diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 0.99, 0.0 , diff.cpu().detach().numpy() ))
    valid_num = np.sum(diff.cpu().detach().numpy() > -1 )

    # output_n = torch.from_numpy(np.where(output_n.cpu()<0.001,0.0,output_n))
    # depth_n = torch.from_numpy(np.where(depth_n.cpu()<0.001,0.0,depth_n))

    sigma1_105_num = np.sum(( diff.cpu().detach().numpy() < 0.1 ) )
    sigma1_110_num = np.sum( (diff.cpu().detach().numpy() > 0.1 )&( diff.cpu().detach().numpy() < 0.3 ) )
    sigma1_125_num = np.sum( (diff.cpu().detach().numpy() > 0.3 )&( diff.cpu().detach().numpy() < 0.5 ) )
    sigma1_225_num = np.sum( (diff.cpu().detach().numpy() > 0.5 )&( diff.cpu().detach().numpy() < 0.7 ) )

    # print(sigma1_105_num/valid_num)
    # print(sigma1_110_num/valid_num)
    # print(sigma1_125_num/valid_num)
    # print(sigma1_225_num/valid_num)

    # sigma1_loss = (maxRatio < 1.05).float().mean().float()
    # sigma2_loss = (maxRatio < 1.10 ).float().mean().float()
    # sigma3_loss = (maxRatio < 100).float().mean().float()
    # print(sigma1_loss)
    # print(sigma2_loss)
    # print(sigma3_loss)
    # plt.subplot(221),plt.imshow(output.cpu().detach().numpy()[0, 0, :, :])
    # plt.subplot(222),plt.imshow(depth.cpu().detach().numpy()[0, 0, :, :])
    # plt.subplot(223),plt.imshow(output_n.cpu().detach().numpy()[0, 0, :, :])
    # plt.subplot(224),plt.imshow(depth_n.cpu().detach().numpy()[0, 0, :, :])
    # plt.show()

    edge = sobel_(output_n)
    pred_edge = sobel_(depth_n)

    # Compute the loss
    l_sobel = nn.L1Loss()(edge, pred_edge)
    l_depth = nn.L1Loss()(output_n, depth_n)
    ssim_value = ssim(output_n, depth_n, val_range=1000.0 / 10.0)
    l_ssim = torch.clamp((1 - ssim_value) * 0.5, 0, 1)

    # if torch.isnan(l_sobel):
    #     print(1)
    #     print(torch.isnan(l_sobel), torch.isnan(l_depth), torch.isnan(l_ssim))

    if cum_num == 4:
        writer.add_scalar('Syn_Known/L1', l_depth.item(), global_step)
        writer.add_scalar('Syn_Known/SSIM', l_ssim.item(), global_step)
        writer.add_scalar('Syn_Known/EDGE', l_sobel.item(), global_step)
        # plt.subplot(221), plt.imshow(image.cpu().detach().numpy()[0, 0, :, :])
        # plt.subplot(222), plt.imshow(depth.cpu().detach().numpy()[0, 0, :, :])
        # plt.subplot(223), plt.imshow(output_n.cpu().detach().numpy()[0, 0, :, :])
        # plt.subplot(224), plt.imshow(output.cpu().detach().numpy()[0, 0, :, :])
        # plt.show()
        # writer.add_image('Train.3.Normal', vutils.make_grid(normal.data, nrow=6, normalize=False), epoch)
        writer.add_image('Syn_Known.1.Depth', colorize(vutils.make_grid((depth).data, nrow=6, normalize=False)), global_step)
        writer.add_image('Syn_Known.2.Ours', colorize(vutils.make_grid((output).data, nrow=6, normalize=False)), global_step)
        writer.add_image('Syn_Known.3.Diff', colorize(vutils.make_grid(torch.abs(output-depth).data, nrow=6, normalize=False)),global_step)
        # writer.add_image('Syn_Known.3.Diff', colorize(vutils.make_grid(torch.abs( image[:, 3: 4,:,:]).data, nrow=6, normalize=False)),

    diff = (output_n - depth_n)
    abs_diff = diff.abs()
    square_diff = torch.pow(abs_diff, 2)
    mae_loss = abs_diff.mean()                         #平均绝对误差：样本绝对误差的绝对值.
    #mae_loss_output_n_depth = (output_n - depth).abs().mean()

    rel_loss = (abs_diff/depth_n).mean()               #平均相对误差
    rmse_loss = torch.pow((square_diff.mean()), 0.5)   #根均方误差：均方误差的算术平方根.
    # maxRatio = torch.max(output_n / depth_n, depth_n / output_n)
    # sigma1_loss = (maxRatio < 1.05).float().mean().float()
    # sigma2_loss = (maxRatio < 1.10 ).float().mean().float()
    # sigma3_loss = (maxRatio < 1.25 ).float().mean().float()
    # print(sigma1_loss)
    # print(sigma2_loss)
    # print(sigma3_loss)
    #f.write('our,%s==>%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n' % (datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"), rel_loss.item(), rmse_loss.item(), log10_loss.item(), sigma1_loss.item(), sigma2_loss.item(), sigma3_loss.item()))
    # print(sigma1_loss.item())
    avg_sigma1_loss_105 = ((sigma1_105_num/valid_num) + last_105 * cum_num) / (cum_num+1)
    # print(avg_sigma1_loss_105)
    if cum_num == 4:
        writer.add_scalar('Syn_Known/θ1.05', avg_sigma1_loss_105, global_step)
        writer.add_scalar('Syn_Known/θ1.10', (sigma1_110_num/valid_num).item(), global_step)
        writer.add_scalar('Syn_Known/θ1.25', (sigma1_125_num/valid_num).item(), global_step)

        writer.add_scalar('Syn_Known/SSIM', ssim_value, global_step)
        writer.add_scalar('Syn_Known/MAE', mae_loss.item(), global_step)
        writer.add_scalar('Syn_Known/REL', rel_loss.item(), global_step)
        writer.add_scalar('Syn_Known/RMSE', rmse_loss.item(), global_step)

    del image
    del depth
    del output
    del edge
    del pred_edge
    return avg_sigma1_loss_105
def LogProgress_test_1(model, writer, test_loader, global_step, cum_num=0, last_105=0):
    model.eval()

    sequential = test_loader
    sample_batched = next(iter(sequential))

    image = torch.autograd.Variable(sample_batched['image'].cuda())
    depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

    # Normalize depth
    # depth_n = DepthNorm(depth)######################

    output, picture = model(image)

    ones_mask = image[:, 6:7, :, :]
    output_n = output * (ones_mask) * 255
    depth_n = depth * (ones_mask) * 255

    diff = (torch.max(output_n / depth_n, depth_n / output_n))
    diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.25, 0.6 , diff.cpu().detach().numpy() ))
    diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.10, 0.4 , diff.cpu().detach().numpy() ))
    diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.05, 0.2 , diff.cpu().detach().numpy() ))
    diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 0.99, 0.0 , diff.cpu().detach().numpy() ))
    valid_num = np.sum(diff.cpu().detach().numpy() > -1 )

    sigma1_105_num = np.sum(( diff.cpu().detach().numpy() < 0.1 ) )
    sigma1_110_num = np.sum( (diff.cpu().detach().numpy() > 0.1 )&( diff.cpu().detach().numpy() < 0.3 ) )
    sigma1_125_num = np.sum( (diff.cpu().detach().numpy() > 0.3 )&( diff.cpu().detach().numpy() < 0.5 ) )
    sigma1_225_num = np.sum( (diff.cpu().detach().numpy() > 0.5 )&( diff.cpu().detach().numpy() < 0.7 ) )

    # print(sigma1_105_num/valid_num)
    # print(sigma1_110_num/valid_num)
    # print(sigma1_125_num/valid_num)
    # print(sigma1_225_num/valid_num)

    edge = sobel_(output_n)
    pred_edge = sobel_(depth_n)

    # Compute the loss
    l_sobel = nn.L1Loss()(edge, pred_edge)
    l_depth = nn.L1Loss()(output_n, depth_n)
    ssim_value = ssim(output_n, depth_n, val_range=1000.0 / 10.0)
    l_ssim = torch.clamp((1 - ssim_value) * 0.5, 0, 1)

    # if torch.isnan(l_sobel):
    #     print(1)
    #     print(torch.isnan(l_sobel), torch.isnan(l_depth), torch.isnan(l_ssim))

    if cum_num == 4:
        writer.add_scalar('Syn_Novel/L1', l_depth.item(), global_step)
        writer.add_scalar('Syn_Novel/SSIM', l_ssim.item(), global_step)
        writer.add_scalar('Syn_Novel/EDGE', l_sobel.item(), global_step)
        # plt.subplot(221), plt.imshow(image.cpu().detach().numpy()[0, 0, :, :])
        # plt.subplot(222), plt.imshow(depth.cpu().detach().numpy()[0, 0, :, :])
        # plt.subplot(223), plt.imshow(output_n.cpu().detach().numpy()[0, 0, :, :])
        # plt.subplot(224), plt.imshow(output.cpu().detach().numpy()[0, 0, :, :])
        # plt.show()
        # writer.add_image('Train.3.Normal', vutils.make_grid(normal.data, nrow=6, normalize=False), epoch)
        writer.add_image('Syn_Novel.1.Depth', colorize(vutils.make_grid((depth).data, nrow=6, normalize=False)), global_step)
        writer.add_image('Syn_Novel.2.Ours', colorize(vutils.make_grid((output).data, nrow=6, normalize=False)), global_step)
        writer.add_image('Syn_Novel.3.Diff', colorize(vutils.make_grid(torch.abs((output - depth)).data, nrow=6, normalize=False)),global_step)
        # writer.add_image('Syn_Known.3.Diff', colorize(vutils.make_grid(torch.abs( image[:, 3: 4,:,:]).data, nrow=6, normalize=False)),


    diff = (output_n - depth_n)
    abs_diff = diff.abs()
    square_diff = torch.pow(abs_diff, 2)
    mae_loss = abs_diff.mean()                         #平均绝对误差：样本绝对误差的绝对值.
    #mae_loss_output_n_depth = (output_n - depth).abs().mean()
    rel_loss = (abs_diff/depth_n).mean()               #平均相对误差
    rmse_loss = torch.pow((square_diff.mean()), 0.5)   #根均方误差：均方误差的算术平方根.

    # maxRatio = torch.max(output_n / depth_n, depth_n / output_n)
    # sigma1_loss = (maxRatio < 1.05).float().mean().float()
    # sigma2_loss = (maxRatio < 1.10 ).float().mean().float()
    # sigma3_loss = (maxRatio < 1.25 ).float().mean().float()
    #f.write('our,%s==>%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n' % (datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"), rel_loss.item(), rmse_loss.item(), log10_loss.item(), sigma1_loss.item(), sigma2_loss.item(), sigma3_loss.item()))

    # print(sigma1_loss.item())
    avg_sigma1_loss_105 = ((sigma1_105_num/valid_num) + last_105 * cum_num) / (cum_num+1)
    # print(avg_sigma1_loss_105)
    if cum_num == 4:
        writer.add_scalar('Syn_Novel/θ1.05', avg_sigma1_loss_105, global_step)
        writer.add_scalar('Syn_Novel/θ1.10', (sigma1_110_num/valid_num).item(), global_step)
        writer.add_scalar('Syn_Novel/θ1.25', (sigma1_125_num/valid_num).item(), global_step)

        writer.add_scalar('Syn_Novel/SSIM', ssim_value, global_step)
        writer.add_scalar('Syn_Novel/MAE', mae_loss.item(), global_step)
        writer.add_scalar('Syn_Novel/REL', rel_loss.item(), global_step)
        writer.add_scalar('Syn_Novel/RMSE', rmse_loss.item(), global_step)

    del image
    del depth
    del output
    del edge
    del pred_edge
    return avg_sigma1_loss_105
def LogProgress_val_real_1(model, writer, test_loader, global_step, cum_num=0,last_105=0):
    model.eval()

    sequential = test_loader
    sample_batched = next(iter(sequential))

    image = torch.autograd.Variable(sample_batched['image'].cuda())
    depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

    # Normalize depth
    # depth_n = DepthNorm(depth)

    output, picture = model(image,is_real = True)

    ones_mask = image[:, 6:7, :, :]
    valid_area = torch.from_numpy(np.where(depth.cpu().detach().numpy() > 0, 1, 0)).cuda()
    output_n = output * (ones_mask) * 255 * valid_area
    depth_n = depth * (ones_mask) * 255* valid_area

    diff = (torch.max(output_n / depth_n, depth_n / output_n))
    diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.25, 0.6, diff.cpu().detach().numpy()))
    diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.10, 0.4, diff.cpu().detach().numpy()))
    diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.05, 0.2, diff.cpu().detach().numpy()))
    diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 0.99, 0.0, diff.cpu().detach().numpy()))
    valid_num = np.sum(diff.cpu().detach().numpy() > -1)

    sigma1_105_num = np.sum((diff.cpu().detach().numpy() < 0.1))
    sigma1_110_num = np.sum((diff.cpu().detach().numpy() > 0.1) & (diff.cpu().detach().numpy() < 0.3))
    sigma1_125_num = np.sum((diff.cpu().detach().numpy() > 0.3) & (diff.cpu().detach().numpy() < 0.5))
    sigma1_225_num = np.sum((diff.cpu().detach().numpy() > 0.5) & (diff.cpu().detach().numpy() < 0.7))

    # print(sigma1_105_num / valid_num)
    # print(sigma1_110_num / valid_num)
    # print(sigma1_125_num / valid_num)
    # print(sigma1_225_num / valid_num)

    edge = sobel_(output_n)
    pred_edge = sobel_(depth_n)

    # Compute the loss
    l_sobel = nn.L1Loss()(edge, pred_edge)
    l_depth = nn.L1Loss()(output_n, depth_n)
    ssim_value = ssim(output_n, depth_n, val_range=1000.0 / 10.0)
    l_ssim = torch.clamp((1 - ssim_value) * 0.5, 0, 1)

    # if torch.isnan(l_sobel):
    #     print(1)
    #     print(torch.isnan(l_sobel), torch.isnan(l_depth), torch.isnan(l_ssim))
    # plt.subplot(222), plt.imshow(diff.cpu().detach().numpy()[0, 0, :, :])
    # plt.subplot(222), plt.imshow(depth.cpu().detach().numpy()[0, 0, :, :])
    # plt.subplot(223), plt.imshow(output_n.cpu().detach().numpy()[0, 0, :, :])
    # plt.subplot(224), plt.imshow(output.cpu().detach().numpy()[0, 0, :, :])
    # plt.show()
    if cum_num == 4:
        writer.add_scalar('Real_Known/L1', l_depth.item(), global_step)
        writer.add_scalar('Real_Known/SSIM', l_ssim.item(), global_step)
        writer.add_scalar('Real_Known/EDGE', l_sobel.item(), global_step)
        # plt.subplot(221), plt.imshow(image.cpu().detach().numpy()[0, 0, :, :])
        # plt.subplot(222), plt.imshow(depth.cpu().detach().numpy()[0, 0, :, :])
        # plt.subplot(223), plt.imshow(output_n.cpu().detach().numpy()[0, 0, :, :])
        # plt.subplot(224), plt.imshow(output.cpu().detach().numpy()[0, 0, :, :])
        # plt.show()
        # writer.add_image('Train.3.Normal', vutils.make_grid(normal.data, nrow=6, normalize=False), epoch)
        writer.add_image('Real_Known.1.Depth', colorize(vutils.make_grid((depth_n).data, nrow=6, normalize=False)),
                         global_step)
        writer.add_image('Real_Known.2.Ours', colorize(vutils.make_grid((output_n).data, nrow=6, normalize=False)),
                         global_step)
        writer.add_image('Real_Known.3.Diff',
                         colorize(vutils.make_grid(torch.abs((output_n - depth_n)).data, nrow=6, normalize=False)),
                         global_step)
        # writer.add_image('Syn_Known.3.Diff', colorize(vutils.make_grid(torch.abs( image[:, 3: 4,:,:]).data, nrow=6, normalize=False)),

    diff = (output_n - depth_n)
    abs_diff = diff.abs()
    square_diff = torch.pow(abs_diff, 2)
    mae_loss = abs_diff.mean()  # 平均绝对误差：样本绝对误差的绝对值.
    # mae_loss_output_n_depth = (output_n - depth).abs().mean()
    rel_loss = (abs_diff / depth_n).mean()  # 平均相对误差
    rmse_loss = torch.pow((square_diff.mean()), 0.5)  # 根均方误差：均方误差的算术平方根.

    # maxRatio = torch.max(output_n / depth_n, depth_n / output_n)
    # sigma1_loss = (maxRatio < 1.05).float().mean().float()
    # sigma2_loss = (maxRatio < 1.10 ).float().mean().float()
    # sigma3_loss = (maxRatio < 1.25 ).float().mean().float()
    # f.write('our,%s==>%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n' % (datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"), rel_loss.item(), rmse_loss.item(), log10_loss.item(), sigma1_loss.item(), sigma2_loss.item(), sigma3_loss.item()))

    # print(sigma1_loss.item())
    avg_sigma1_loss_105 = ((sigma1_105_num / valid_num) + last_105 * cum_num) / (cum_num + 1)
    # print(avg_sigma1_loss_105)
    if cum_num == 4:
        writer.add_scalar('Real_Known/θ1.05', avg_sigma1_loss_105, global_step)
        writer.add_scalar('Real_Known/θ1.10', (sigma1_110_num / valid_num).item(), global_step)
        writer.add_scalar('Real_Known/θ1.25', (sigma1_125_num / valid_num).item(), global_step)

        writer.add_scalar('Real_Known/SSIM', ssim_value, global_step)
        writer.add_scalar('Real_Known/MAE', mae_loss.item(), global_step)
        writer.add_scalar('Real_Known/REL', rel_loss.item(), global_step)
        writer.add_scalar('Real_Known/RMSE', rmse_loss.item(), global_step)

    del image
    del depth
    del output
    del edge
    del pred_edge
    return avg_sigma1_loss_105
def LogProgress_test_real_1(model, writer, test_loader, global_step, cum_num=0,last_105=0):
    model.eval()

    sequential = test_loader
    sample_batched = next(iter(sequential))

    image = torch.autograd.Variable(sample_batched['image'].cuda())
    depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

    # Normalize depth
    # depth_n = DepthNorm(depth)######################

    output, picture = model(image,is_real = True)

    ones_mask = image[:, 6:7, :, :]
    valid_area = torch.from_numpy(np.where(depth.cpu().detach().numpy() > 0, 1, 0)).cuda()
    output_n = output * (ones_mask) * 255 * valid_area
    depth_n = depth * (ones_mask) * 255 * valid_area

    diff = (torch.max(output_n / depth_n, depth_n / output_n))
    diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.25, 0.6, diff.cpu().detach().numpy()))
    diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.10, 0.4, diff.cpu().detach().numpy()))
    diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.05, 0.2, diff.cpu().detach().numpy()))
    diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 0.99, 0.0, diff.cpu().detach().numpy()))
    valid_num = np.sum(diff.cpu().detach().numpy() > -1)

    sigma1_105_num = np.sum((diff.cpu().detach().numpy() < 0.1))
    sigma1_110_num = np.sum((diff.cpu().detach().numpy() > 0.1) & (diff.cpu().detach().numpy() < 0.3))
    sigma1_125_num = np.sum((diff.cpu().detach().numpy() > 0.3) & (diff.cpu().detach().numpy() < 0.5))
    sigma1_225_num = np.sum((diff.cpu().detach().numpy() > 0.5) & (diff.cpu().detach().numpy() < 0.7))

    # print(sigma1_105_num / valid_num)
    # print(sigma1_110_num / valid_num)
    # print(sigma1_125_num / valid_num)
    # print(sigma1_225_num / valid_num)
    # plt.subplot(222), plt.imshow(diff.cpu().detach().numpy()[0, 0, :, :])
    # plt.show()
    edge = sobel_(output_n)
    pred_edge = sobel_(depth_n)

    # Compute the loss
    l_sobel = nn.L1Loss()(edge, pred_edge)
    l_depth = nn.L1Loss()(output_n, depth_n)
    ssim_value = ssim(output_n, depth_n, val_range=1000.0 / 10.0)
    l_ssim = torch.clamp((1 - ssim_value) * 0.5, 0, 1)

    # if torch.isnan(l_sobel):
    #     print(1)
    #     print(torch.isnan(l_sobel), torch.isnan(l_depth), torch.isnan(l_ssim))

    if cum_num == 4:
        writer.add_scalar('Real_Novel/L1', l_depth.item(), global_step)
        writer.add_scalar('Real_Novel/SSIM', l_ssim.item(), global_step)
        writer.add_scalar('Real_Novel/EDGE', l_sobel.item(), global_step)
        # plt.subplot(221), plt.imshow(image.cpu().detach().numpy()[0, 0, :, :])
        # plt.subplot(222), plt.imshow(depth.cpu().detach().numpy()[0, 0, :, :])
        # plt.subplot(223), plt.imshow(output_n.cpu().detach().numpy()[0, 0, :, :])
        # plt.subplot(224), plt.imshow(output.cpu().detach().numpy()[0, 0, :, :])
        # plt.show()
        # writer.add_image('Train.3.Normal', vutils.make_grid(normal.data, nrow=6, normalize=False), epoch)
        writer.add_image('Real_Novel.1.Depth', colorize(vutils.make_grid((depth_n).data, nrow=6, normalize=False)),
                         global_step)
        writer.add_image('Real_Novel.2.Ours', colorize(vutils.make_grid((output_n).data, nrow=6, normalize=False)),
                         global_step)
        writer.add_image('Real_Novel.3.Diff',
                         colorize(vutils.make_grid(torch.abs((output_n - depth_n)).data, nrow=6, normalize=False)),
                         global_step)
        # writer.add_image('Syn_Known.3.Diff', colorize(vutils.make_grid(torch.abs( image[:, 3: 4,:,:]).data, nrow=6, normalize=False)),

    diff = (output_n - depth_n)
    abs_diff = diff.abs()
    square_diff = torch.pow(abs_diff, 2)
    mae_loss = abs_diff.mean()  # 平均绝对误差：样本绝对误差的绝对值.
    # mae_loss_output_n_depth = (output_n - depth).abs().mean()
    rel_loss = (abs_diff / depth_n).mean()  # 平均相对误差
    rmse_loss = torch.pow((square_diff.mean()), 0.5)  # 根均方误差：均方误差的算术平方根.

    # maxRatio = torch.max(output_n / depth_n, depth_n / output_n)
    # sigma1_loss = (maxRatio < 1.05).float().mean().float()
    # sigma2_loss = (maxRatio < 1.10 ).float().mean().float()
    # sigma3_loss = (maxRatio < 1.25 ).float().mean().float()
    # f.write('our,%s==>%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n' % (datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"), rel_loss.item(), rmse_loss.item(), log10_loss.item(), sigma1_loss.item(), sigma2_loss.item(), sigma3_loss.item()))

    # print(sigma1_loss.item())
    avg_sigma1_loss_105 = ((sigma1_105_num / valid_num) + last_105 * cum_num) / (cum_num + 1)
    # print(avg_sigma1_loss_105)
    if cum_num == 4:
        writer.add_scalar('Real_Novel/θ1.05', avg_sigma1_loss_105, global_step)
        writer.add_scalar('Real_Novel/θ1.10', (sigma1_110_num / valid_num).item(), global_step)
        writer.add_scalar('Real_Novel/θ1.25', (sigma1_125_num / valid_num).item(), global_step)

        writer.add_scalar('Real_Novel/SSIM', ssim_value, global_step)
        writer.add_scalar('Real_Novel/MAE', mae_loss.item(), global_step)
        writer.add_scalar('Real_Novel/REL', rel_loss.item(), global_step)
        writer.add_scalar('Real_Novel/RMSE', rmse_loss.item(), global_step)

    del image
    del depth
    del output
    del edge
    del pred_edge
    return avg_sigma1_loss_105



def LogProgress_val(model, writer, test_loader, global_step, cum_num=0, last_105=0):
    model.eval()
    sigma1_105_num_sum = 0
    sigma1_110_num_sum = 0
    sigma1_125_num_sum = 0

    valid_num_sum = 0
    typeo_of_dataset = 'Syn_Known'
    l_depth_sum = 0
    l_sobel_sum = 0
    mae_sum = 0
    rmse_sum = 0
    rel_sum = 0
    for i, sample_batched in enumerate(test_loader):

        image = torch.autograd.Variable(sample_batched['image'].cuda())
        depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
        # plt.imshow(depth.cpu()[0, 0, :, :])
        # plt.show()

        output, picture = model(image)
        ones_mask = image[:, 6:7, :, :]

        output_n = output * (ones_mask) * 255
        depth_n = depth * (ones_mask) * 255

        # maxRatio = torch.max(output_n / depth_n, depth_n / output_n)
        diff = (torch.max(output_n / depth_n, depth_n / output_n))
        diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.25, 0.6 , diff.cpu().detach().numpy() ))
        diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.10, 0.4 , diff.cpu().detach().numpy() ))
        diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.05, 0.2 , diff.cpu().detach().numpy() ))
        diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 0.99, 0.0 , diff.cpu().detach().numpy() ))
        valid_num = np.sum(diff.cpu().detach().numpy() > -1 )

        # output_n = torch.from_numpy(np.where(output_n.cpu()<0.001,0.0,output_n))
        # depth_n = torch.from_numpy(np.where(depth_n.cpu()<0.001,0.0,depth_n))

        sigma1_105_num = np.sum(( diff.cpu().detach().numpy() < 0.1 ) )
        sigma1_110_num = np.sum( (diff.cpu().detach().numpy() > 0.1 )&( diff.cpu().detach().numpy() < 0.3 ) )
        sigma1_125_num = np.sum( (diff.cpu().detach().numpy() > 0.3 )&( diff.cpu().detach().numpy() < 0.5 ) )
        sigma1_225_num = np.sum( (diff.cpu().detach().numpy() > 0.5 )&( diff.cpu().detach().numpy() < 0.7 ) )
        sigma1_105_num_sum = sigma1_105_num_sum + sigma1_105_num
        sigma1_110_num_sum = sigma1_110_num_sum + sigma1_110_num + sigma1_105_num
        sigma1_125_num_sum = sigma1_125_num_sum + sigma1_105_num + sigma1_110_num + sigma1_125_num
        valid_num_sum = valid_num_sum + valid_num
        edge = sobel_(output_n)
        pred_edge = sobel_(depth_n)
        # rmse = ((gt - pred)**2).mean().sqrt()
        # rmse_log = ((safe_log(gt) - safe_log(pred))**2).mean().sqrt()
        # log10 = (safe_log10(gt) - safe_log10(pred)).abs().mean()
        # abs_rel = ((gt - pred).abs() / gt).mean()
        # mae = (gt - pred).abs().mean()
        # sq_rel = ((gt - pred)**2 / gt).mean()

        # Compute the loss
        # l_sobel = nn.L1Loss()(edge, pred_edge)
        # l_depth = nn.L1Loss()(output_n, depth_n)
        # ssim_value = ssim(output_n, depth_n, val_range=1000.0 / 10.0)
        # l_ssim = torch.clamp((1 - ssim_value) * 0.5, 0, 1)

        # l_depth_sum = l_depth_sum + l_depth.item()*64
        # l_sobel_sum = l_sobel_sum + l_sobel.item()*64
        del image
        del depth
        del output
        del diff
        del edge
        del pred_edge
        depth_n =  depth_n * 255 /100
        output_n = output_n * 255 / 100
        depth_n = (depth_n.cpu().detach().numpy())
        output_n = (output_n.cpu().detach().numpy())

        diffen = (depth_n - output_n)
        mae_sum = mae_sum + np.sum(np.abs(diffen))
        rmse_sum = rmse_sum + np.sum((diffen)**2)


        depth_n = np.where(depth_n==0,1e-4,depth_n)
        output_n = np.where(output_n == 0, 1e-4, output_n)

        rel_sum = rel_sum + np.sum(np.abs( ((depth_n - output_n)/depth_n) ))
        del output_n
        del depth_n
    mae_loss = mae_sum / 532 / 128 / 256
    rel_loss = rel_sum / 532 / 128 / 256
    rmse_loss = np.sqrt((rmse_sum / 532 / 128 / 256))



    writer.add_scalar(typeo_of_dataset + '/MAE', mae_loss, global_step)
    writer.add_scalar(typeo_of_dataset + '/RMSE', rmse_loss, global_step)
    writer.add_scalar(typeo_of_dataset + '/REL', rel_loss, global_step)
    # writer.add_scalar(typeo_of_dataset+'/MAE', mae_loss, global_step)
    # writer.add_scalar('Syn_Known/REL', rel_loss.item(), global_step)
    # writer.add_scalar('Syn_Known/RMSE', rmse_loss.item(), global_step)
        # if torch.isnan(l_sobel):
        #     print(1)
        #     print(torch.isnan(l_sobel), torch.isnan(l_depth), torch.isnan(l_ssim))

   
    writer.add_scalar(typeo_of_dataset+'/θ1.05', (sigma1_105_num_sum/valid_num_sum).item(), global_step)
    writer.add_scalar(typeo_of_dataset+'/θ1.10', (sigma1_110_num_sum/valid_num_sum).item(), global_step)
    writer.add_scalar(typeo_of_dataset+'/θ1.25', (sigma1_125_num_sum/valid_num_sum).item(),global_step)

        # writer.add_scalar('Syn_Known/SSIM', ssim_value, global_step)

    return (sigma1_105_num_sum/valid_num_sum)
def LogProgress_test(model, writer, test_loader, global_step, cum_num=0, last_105=0):
    model.eval()
    sigma1_105_num_sum = 0
    sigma1_110_num_sum = 0
    sigma1_125_num_sum = 0

    valid_num_sum = 0
    typeo_of_dataset = 'Syn_Novel'

    mae_sum = 0
    rmse_sum = 0
    rel_sum = 0
    for i, sample_batched in enumerate(test_loader):

        image = torch.autograd.Variable(sample_batched['image'].cuda())
        depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
        # len_num = len(depth)
        # Normalize depth
        # depth_n = DepthNorm(depth)######################

        output, picture = model(image)
        ones_mask = image[:, 6:7, :, :]
        #
        output_n = output * (ones_mask) * 255
        depth_n = depth * (ones_mask) * 255

        # maxRatio = torch.max(output_n / depth_n, depth_n / output_n)
        diff = (torch.max(output_n / depth_n, depth_n / output_n))
        diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.25, 0.6 , diff.cpu().detach().numpy() ))
        diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.10, 0.4 , diff.cpu().detach().numpy() ))
        diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.05, 0.2 , diff.cpu().detach().numpy() ))
        diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 0.99, 0.0 , diff.cpu().detach().numpy() ))
        valid_num = np.sum(diff.cpu().detach().numpy() > -1 )

        # output_n = torch.from_numpy(np.where(output_n.cpu()<0.001,0.0,output_n))
        # depth_n = torch.from_numpy(np.where(depth_n.cpu()<0.001,0.0,depth_n))

        sigma1_105_num = np.sum(( diff.cpu().detach().numpy() < 0.1 ) )
        sigma1_110_num = np.sum( (diff.cpu().detach().numpy() > 0.1 )&( diff.cpu().detach().numpy() < 0.3 ) )
        sigma1_125_num = np.sum( (diff.cpu().detach().numpy() > 0.3 )&( diff.cpu().detach().numpy() < 0.5 ) )
        sigma1_225_num = np.sum( (diff.cpu().detach().numpy() > 0.5 )&( diff.cpu().detach().numpy() < 0.7 ) )


        sigma1_105_num_sum = sigma1_105_num_sum + sigma1_105_num
        sigma1_110_num_sum = sigma1_110_num_sum + sigma1_110_num + sigma1_105_num
        sigma1_125_num_sum = sigma1_125_num_sum + sigma1_105_num + sigma1_110_num + sigma1_125_num
        valid_num_sum = valid_num_sum + valid_num
        # print(sigma1_105_num/valid_num)
        # print(sigma1_110_num/valid_num)
        # print(sigma1_125_num/valid_num)
        # print(sigma1_225_num/valid_num)

        # sigma1_loss = (maxRatio < 1.05).float().mean().float()
        # sigma2_loss = (maxRatio < 1.10 ).float().mean().float()
        # sigma3_loss = (maxRatio < 100).float().mean().float()
        # print(sigma1_loss)
        # print(sigma2_loss)
        # print(sigma3_loss)
        # plt.subplot(221),plt.imshow(output.cpu().detach().numpy()[0, 0, :, :])
        # plt.subplot(222),plt.imshow(depth.cpu().detach().numpy()[0, 0, :, :])
        # plt.subplot(223),plt.imshow(output_n.cpu().detach().numpy()[0, 0, :, :])
        # plt.subplot(224),plt.imshow(depth_n.cpu().detach().numpy()[0, 0, :, :])
        # plt.show()

        edge = sobel_(output_n)
        pred_edge = sobel_(depth_n)

        # Compute the loss
        # l_sobel = nn.L1Loss()(edge, pred_edge)
        # l_depth = nn.L1Loss()(output_n, depth_n)
        # ssim_value = ssim(output_n, depth_n, val_range=1000.0 / 10.0)
        # l_ssim = torch.clamp((1 - ssim_value) * 0.5, 0, 1)
        # l_sobel_sum = l_sobel_sum + l_sobel * len_num
        # l_depth_sum = l_depth_sum + l_depth * len_num
        # l_depth_sum = l_depth_sum + l_depth.item()*64
        # l_sobel_sum = l_sobel_sum + l_sobel.item()*64
        del image
        del depth
        del output
        del diff
        del edge
        del pred_edge
        depth_n =  depth_n * 255 /100
        output_n = output_n * 255 / 100
        depth_n = (depth_n.cpu().detach().numpy())
        output_n = (output_n.cpu().detach().numpy())

        diffen = (depth_n - output_n)
        mae_sum = mae_sum + np.sum(np.abs(diffen))
        rmse_sum = rmse_sum + np.sum((diffen)**2)

        depth_n = np.where(depth_n==0,1e-4,depth_n)
        output_n = np.where(output_n == 0, 1e-4, output_n)

        rel_sum = rel_sum + np.sum(np.abs( ((depth_n - output_n)/depth_n) ))
        del depth_n
        del output_n

        # if torch.isnan(l_sobel):
        #     print(1)
        #     print(torch.isnan(l_sobel), torch.isnan(l_depth), torch.isnan(l_ssim))

  
    writer.add_scalar(typeo_of_dataset+'/θ1.05', (sigma1_105_num_sum/valid_num_sum).item(), global_step)
    writer.add_scalar(typeo_of_dataset+'/θ1.10', (sigma1_110_num_sum/valid_num_sum).item(), global_step)
    writer.add_scalar(typeo_of_dataset+'/θ1.25', (sigma1_125_num_sum/valid_num_sum).item(),global_step)
        #
        # writer.add_scalar('Syn_Known/SSIM', ssim_value, global_step)
        # writer.add_scalar('Syn_Known/MAE', mae_loss.item(), global_step)
        # writer.add_scalar('Syn_Known/REL', rel_loss.item(), global_step)
        # writer.add_scalar('Syn_Known/RMSE', rmse_loss.item(), global_step)

    return (sigma1_105_num_sum/valid_num_sum)

def LogProgress_val_real(model, writer, test_loader, global_step, cum_num=0, last_105=0):
    model.eval()
    sigma1_105_num_sum = 0
    sigma1_110_num_sum = 0
    sigma1_125_num_sum = 0

    valid_num_sum = 0
    typeo_of_dataset='Real_Known'
    l_sobel_sum = 0
    l_depth_sum = 0
    mae_sum = 0
    rmse_sum = 0
    rel_sum = 0
    for i, sample_batched in enumerate(test_loader):

        image = torch.autograd.Variable(sample_batched['image'].cuda())
        depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
        # plt.imshow(depth.cpu()[0,0,:,:])
        # plt.show()
        # len_num = len(depth)
        # Normalize depth
        # depth_n = DepthNorm(depth)######################
        output, picture = model(image, is_real=True)
        ones_mask = image[:, 6:7, :, :]
        valid_area = torch.from_numpy(np.where(depth.cpu().detach().numpy() > 0, 1, 0)).cuda()
        output_n = output * (ones_mask) * 255 * valid_area
        depth_n = depth * (ones_mask) * 255 * valid_area

        # maxRatio = torch.max(output_n / depth_n, depth_n / output_n)
        diff = (torch.max(output_n / depth_n, depth_n / output_n))
        diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.25, 0.6 , diff.cpu().detach().numpy() ))
        diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.10, 0.4 , diff.cpu().detach().numpy() ))
        diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.05, 0.2 , diff.cpu().detach().numpy() ))
        diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 0.99, 0.0 , diff.cpu().detach().numpy() ))
        valid_num = np.sum(diff.cpu().detach().numpy() > -1 )

        # output_n = torch.from_numpy(np.where(output_n.cpu()<0.001,0.0,output_n))
        # depth_n = torch.from_numpy(np.where(depth_n.cpu()<0.001,0.0,depth_n))

        sigma1_105_num = np.sum(( diff.cpu().detach().numpy() < 0.1 ) )
        sigma1_110_num = np.sum( (diff.cpu().detach().numpy() > 0.1 )&( diff.cpu().detach().numpy() < 0.3 ) )
        sigma1_125_num = np.sum( (diff.cpu().detach().numpy() > 0.3 )&( diff.cpu().detach().numpy() < 0.5 ) )
        sigma1_225_num = np.sum( (diff.cpu().detach().numpy() > 0.5 )&( diff.cpu().detach().numpy() < 0.7 ) )


        sigma1_105_num_sum = sigma1_105_num_sum + sigma1_105_num
        sigma1_110_num_sum = sigma1_110_num_sum + sigma1_110_num + sigma1_105_num
        sigma1_125_num_sum = sigma1_125_num_sum + sigma1_105_num + sigma1_110_num + sigma1_125_num
        valid_num_sum = valid_num_sum + valid_num
        # print(sigma1_105_num/valid_num)
        # print(sigma1_110_num/valid_num)
        # print(sigma1_125_num/valid_num)
        # print(sigma1_225_num/valid_num)

        # sigma1_loss = (maxRatio < 1.05).float().mean().float()
        # sigma2_loss = (maxRatio < 1.10 ).float().mean().float()
        # sigma3_loss = (maxRatio < 100).float().mean().float()
        # print(sigma1_loss)
        # print(sigma2_loss)
        # print(sigma3_loss)
        # plt.subplot(221),plt.imshow(output.cpu().detach().numpy()[0, 0, :, :])
        # plt.subplot(222),plt.imshow(depth.cpu().detach().numpy()[0, 0, :, :])
        # plt.subplot(223),plt.imshow(output_n.cpu().detach().numpy()[0, 0, :, :])
        # plt.subplot(224),plt.imshow(depth_n.cpu().detach().numpy()[0, 0, :, :])
        # plt.show()

        edge = sobel_(output_n)
        pred_edge = sobel_(depth_n)

 
        del image
        del depth
        del output
        del diff
        del edge
        del pred_edge
        depth_n =  depth_n * 255 /100
        output_n = output_n * 255 / 100
        depth_n = (depth_n.cpu().detach().numpy())
        output_n = (output_n.cpu().detach().numpy())

        diffen = (depth_n - output_n)
        mae_sum = mae_sum + np.sum(np.abs(diffen))
        rmse_sum = rmse_sum + np.sum((diffen)**2)


        depth_n = np.where(depth_n==0,1e-4,depth_n)
        output_n = np.where(output_n == 0, 1e-4, output_n)

        rel_sum = rel_sum + np.sum(np.abs( ((depth_n - output_n)/depth_n) ))
        del output_n
        del depth_n
    mae_loss = mae_sum / 173 / 128 / 256
    rel_loss = rel_sum / 173 / 128 / 256
    rmse_loss = np.sqrt((rmse_sum / 173 / 128 / 256))

    writer.add_scalar(typeo_of_dataset + '/MAE', mae_loss, global_step)
    writer.add_scalar(typeo_of_dataset + '/RMSE', rmse_loss, global_step)
    writer.add_scalar(typeo_of_dataset + '/REL', rel_loss, global_step)
   
    writer.add_scalar(typeo_of_dataset+'/θ1.05', (sigma1_105_num_sum/valid_num_sum).item(), global_step)
    writer.add_scalar(typeo_of_dataset+'/θ1.10', (sigma1_110_num_sum/valid_num_sum).item(), global_step)
    writer.add_scalar(typeo_of_dataset+'/θ1.25', (sigma1_125_num_sum/valid_num_sum).item(),global_step)
    return (sigma1_105_num_sum/valid_num_sum)

def LogProgress_test_real(model, writer, test_loader, global_step, cum_num=0, last_105=0):
    model.eval()
    sigma1_105_num_sum = 0
    sigma1_110_num_sum = 0
    sigma1_125_num_sum = 0
    valid_num_sum = 0
    typeo_of_dataset = 'Real_Novel'
    l_sobel_sum = 0
    l_depth_sum = 0
    mae_sum = 0
    rmse_sum = 0
    rel_sum = 0
    for i, sample_batched in enumerate(test_loader):

        image = torch.autograd.Variable(sample_batched['image'].cuda())
        depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
        # len_num = len(depth)
        # Normalize depth
        # depth_n = DepthNorm(depth)######################

        output, picture = model(image, is_real=True)
        ones_mask = image[:, 6:7, :, :]
        valid_area = torch.from_numpy(np.where(depth.cpu().detach().numpy() > 0, 1, 0)).cuda()
        output_n = output * (ones_mask) * 255 * valid_area
        depth_n = depth * (ones_mask) * 255 * valid_area

        # output_n = output * (ones_mask) * 255
        # depth_n = depth * (ones_mask) * 255

        # maxRatio = torch.max(output_n / depth_n, depth_n / output_n)
        diff = (torch.max(output_n / depth_n, depth_n / output_n))
        diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.25, 0.6 , diff.cpu().detach().numpy() ))
        diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.10, 0.4 , diff.cpu().detach().numpy() ))
        diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.05, 0.2 , diff.cpu().detach().numpy() ))
        diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 0.99, 0.0 , diff.cpu().detach().numpy() ))
        valid_num = np.sum(diff.cpu().detach().numpy() > -1 )

        # output_n = torch.from_numpy(np.where(output_n.cpu()<0.001,0.0,output_n))
        # depth_n = torch.from_numpy(np.where(depth_n.cpu()<0.001,0.0,depth_n))

        sigma1_105_num = np.sum(( diff.cpu().detach().numpy() < 0.1 ) )
        sigma1_110_num = np.sum( (diff.cpu().detach().numpy() > 0.1 )&( diff.cpu().detach().numpy() < 0.3 ) )
        sigma1_125_num = np.sum( (diff.cpu().detach().numpy() > 0.3 )&( diff.cpu().detach().numpy() < 0.5 ) )
        sigma1_225_num = np.sum( (diff.cpu().detach().numpy() > 0.5 )&( diff.cpu().detach().numpy() < 0.7 ) )


        sigma1_105_num_sum = sigma1_105_num_sum + sigma1_105_num
        sigma1_110_num_sum = sigma1_110_num_sum + sigma1_110_num + sigma1_105_num
        sigma1_125_num_sum = sigma1_125_num_sum + sigma1_105_num + sigma1_110_num + sigma1_125_num
        valid_num_sum = valid_num_sum + valid_num

        # print(sigma1_105_num_sum,sigma1_110_num_sum,sigma1_125_num_sum,valid_num_sum)
        # print(sigma1_105_num/valid_num)
        # print(sigma1_110_num/valid_num)
        # print(sigma1_125_num/valid_num)
        # print(sigma1_225_num/valid_num)

        # sigma1_loss = (maxRatio < 1.05).float().mean().float()
        # sigma2_loss = (maxRatio < 1.10 ).float().mean().float()
        # sigma3_loss = (maxRatio < 100).float().mean().float()
        # print(sigma1_loss)
        # print(sigma2_loss)
        # print(sigma3_loss)
        # plt.subplot(221),plt.imshow(output.cpu().detach().numpy()[0, 0, :, :])
        # plt.subplot(222),plt.imshow(depth.cpu().detach().numpy()[0, 0, :, :])
        # plt.subplot(223),plt.imshow(output_n.cpu().detach().numpy()[0, 0, :, :])
        # plt.subplot(224),plt.imshow(depth_n.cpu().detach().numpy()[0, 0, :, :])
        # plt.show()

        edge = sobel_(output_n)
        pred_edge = sobel_(depth_n)

        # Compute the loss
        # l_sobel = nn.L1Loss()(edge, pred_edge)
        # l_depth = nn.L1Loss()(output_n, depth_n)
        # ssim_value = ssim(output_n, depth_n, val_range=1000.0 / 10.0)
        # l_ssim = torch.clamp((1 - ssim_value) * 0.5, 0, 1)
        # l_depth_sum = l_depth_sum + l_depth.item()*64
        # l_sobel_sum = l_sobel_sum + l_sobel.item()*64

        # l_sobel_sum = l_sobel_sum + l_sobel * len_num
        # l_depth_sum = l_depth_sum + l_depth * len_num

        # if torch.isnan(l_sobel):
        #     print(1)
        #     print(torch.isnan(l_sobel), torch.isnan(l_depth), torch.isnan(l_ssim))

  

    # print(avg_sigma1_loss_105)
    # if cum_num == 4:
        del image
        del depth
        del output
        del diff
        del edge
        del pred_edge
        depth_n = depth_n * 255 / 100
        output_n = output_n * 255 / 100
        depth_n = (depth_n.cpu().detach().numpy())
        output_n = (output_n.cpu().detach().numpy())

        diffen = (depth_n - output_n)
        mae_sum = mae_sum + np.sum(np.abs(diffen))
        rmse_sum = rmse_sum + np.sum((diffen) ** 2)

        depth_n = np.where(depth_n == 0, 1e-4, depth_n)
        output_n = np.where(output_n == 0, 1e-4, output_n)

        rel_sum = rel_sum + np.sum(np.abs(((depth_n - output_n) / depth_n)))
        del output_n
        del depth_n
    mae_loss = mae_sum / 113 / 128 / 256
    rel_loss = rel_sum / 113 / 128 / 256
    rmse_loss = np.sqrt((rmse_sum / 113 / 128 / 256))

    writer.add_scalar(typeo_of_dataset + '/MAE', mae_loss, global_step)
    writer.add_scalar(typeo_of_dataset + '/RMSE', rmse_loss, global_step)
    writer.add_scalar(typeo_of_dataset + '/REL', rel_loss, global_step)
    # l_depth_sum = l_depth_sum / 113
    # l_sobel_sum = l_sobel_sum / 113
    # writer.add_scalar(typeo_of_dataset+'/L1', l_depth_sum, global_step)
    # writer.add_scalar(typeo_of_dataset+'/SSIM', l_ssim.item(), global_step)
    # writer.add_scalar(typeo_of_dataset+'/EDGE', l_sobel_sum, global_step)
    writer.add_scalar(typeo_of_dataset+'/θ1.05', (sigma1_105_num_sum/valid_num_sum).item(), global_step)
    writer.add_scalar(typeo_of_dataset+'/θ1.10', (sigma1_110_num_sum/valid_num_sum).item(), global_step)
    writer.add_scalar(typeo_of_dataset+'/θ1.25', (sigma1_125_num_sum/valid_num_sum).item(),global_step)
        #
        # writer.add_scalar('Syn_Known/SSIM', ssim_value, global_step)
        # writer.add_scalar('Syn_Known/MAE', mae_loss.item(), global_step)
        # writer.add_scalar('Syn_Known/REL', rel_loss.item(), global_step)
        # writer.add_scalar('Syn_Known/RMSE', rmse_loss.item(), global_step)

    return (sigma1_105_num_sum/valid_num_sum)

def LogProgress_additional_val(model, writer, test_loader, global_step, cum_num=0, last_105=0):
    model.eval()
    sigma1_105_num_sum = 0
    sigma1_110_num_sum = 0
    sigma1_125_num_sum = 0

    valid_num_sum = 0
    typeo_of_dataset = 'Val_Dataset'

    mae_sum = 0
    rmse_sum = 0
    rel_sum = 0
    loss_l1 = AverageMeter()
    loss_ssim = AverageMeter()
    loss_edge = AverageMeter()
    for i, sample_batched in enumerate(test_loader):

        image = torch.autograd.Variable(sample_batched['image'].cuda())
        depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))


        output, picture = model(image)
        ones_mask = image[:, 6:7, :, :]

        output_n = output * (ones_mask) * 255
        depth_n = depth * (ones_mask) * 255

        # maxRatio = torch.max(output_n / depth_n, depth_n / output_n)
        diff = (torch.max(output_n / depth_n, depth_n / output_n))
        diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.25, 0.6 , diff.cpu().detach().numpy() ))
        diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.10, 0.4 , diff.cpu().detach().numpy() ))
        diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 1.05, 0.2 , diff.cpu().detach().numpy() ))
        diff = torch.from_numpy(np.where(diff.cpu().detach().numpy() > 0.99, 0.0 , diff.cpu().detach().numpy() ))
        valid_num = np.sum(diff.cpu().detach().numpy() > -1 )

        # output_n = torch.from_numpy(np.where(output_n.cpu()<0.001,0.0,output_n))
        # depth_n = torch.from_numpy(np.where(depth_n.cpu()<0.001,0.0,depth_n))

        sigma1_105_num = np.sum(( diff.cpu().detach().numpy() < 0.1 ) )
        sigma1_110_num = np.sum( (diff.cpu().detach().numpy() > 0.1 )&( diff.cpu().detach().numpy() < 0.3 ) )
        sigma1_125_num = np.sum( (diff.cpu().detach().numpy() > 0.3 )&( diff.cpu().detach().numpy() < 0.5 ) )
        sigma1_225_num = np.sum( (diff.cpu().detach().numpy() > 0.5 )&( diff.cpu().detach().numpy() < 0.7 ) )

        sigma1_105_num_sum = sigma1_105_num_sum + sigma1_105_num
        sigma1_110_num_sum = sigma1_110_num_sum + sigma1_110_num + sigma1_105_num
        sigma1_125_num_sum = sigma1_125_num_sum + sigma1_105_num + sigma1_110_num + sigma1_125_num
        valid_num_sum = valid_num_sum + valid_num

        edge = sobel_(output_n)
        pred_edge = sobel_(depth_n)

        # Compute the loss
        l_sobel = nn.L1Loss()(edge, pred_edge)
        l_depth = nn.L1Loss()(output_n, depth_n)
        ssim_value = ssim(output_n, depth_n, val_range=1000.0 / 10.0)
        l_ssim = torch.clamp((1 - ssim_value) * 0.5, 0, 1)

        loss_l1.update(l_depth.data.item(), image.size(0))
        loss_ssim.update(l_ssim.data.item(), image.size(0))
        loss_edge.update(l_sobel.data.item(), image.size(0))

        del image
        del depth
        del output
        del diff
        del edge
        del pred_edge

        depth_n = depth_n * 255 / 100
        output_n = output_n * 255 / 100
        depth_n = (depth_n.cpu().detach().numpy())
        output_n = (output_n.cpu().detach().numpy())

        diffen = (depth_n - output_n)
        mae_sum = mae_sum + np.sum(np.abs(diffen))
        rmse_sum = rmse_sum + np.sum((diffen) ** 2)

        depth_n = np.where(depth_n == 0, 1e-4, depth_n)
        output_n = np.where(output_n == 0, 1e-4, output_n)

        rel_sum = rel_sum + np.sum(np.abs(((depth_n - output_n) / depth_n)))
        del output_n
        del depth_n
    mae_loss = mae_sum / 4498 / 128 / 256
    rel_loss = rel_sum / 4498 / 128 / 256
    rmse_loss = np.sqrt((rmse_sum / 4498 / 128 / 256))
    writer.add_scalar(typeo_of_dataset + '/L1', loss_l1.val, global_step)
    writer.add_scalar(typeo_of_dataset + '/SSIM', loss_ssim.val, global_step)
    writer.add_scalar(typeo_of_dataset + '/EDGE', loss_edge.val, global_step)
    writer.add_scalar(typeo_of_dataset + '/MAE', mae_loss, global_step)
    writer.add_scalar(typeo_of_dataset + '/RMSE', rmse_loss, global_step)
    writer.add_scalar(typeo_of_dataset + '/REL', rel_loss, global_step)

    writer.add_scalar(typeo_of_dataset+'/θ1.05', (sigma1_105_num_sum/valid_num_sum).item(), global_step)
    writer.add_scalar(typeo_of_dataset+'/θ1.10', (sigma1_110_num_sum/valid_num_sum).item(), global_step)
    writer.add_scalar(typeo_of_dataset+'/θ1.25', (sigma1_125_num_sum/valid_num_sum).item(),global_step)

    return (sigma1_105_num_sum/valid_num_sum)


if __name__ == '__main__':
    main()