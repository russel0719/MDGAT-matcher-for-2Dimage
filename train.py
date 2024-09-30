#encoding: utf-8
from pathlib import Path
import argparse
import torch
from torch.autograd import Variable
import torch.multiprocessing
from tqdm import tqdm
import time
import cv2
import numpy as np
import matplotlib.cm as cm
from tensorboardX import SummaryWriter
from models.superglue import SuperGlue
from models.mdgat import MDGAT
torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')


parser = argparse.ArgumentParser(
    description='Point cloud matching training ',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations')

parser.add_argument(
    '--learning_rate', type=int, default=0.0001,  #0.0001
    help='Learning rate')

parser.add_argument(
    '--epoch', type=int, default=100,
    help='Number of epoches')

parser.add_argument(
    '--memory_is_enough', type=bool, default=True, 
    help='If memory is enough, load all the data')
        
parser.add_argument(
    '--batch_size', type=int, default=1, #12
    help='Batch size')

parser.add_argument(
    '--local_rank', type=int, default=[0,1,2,3], 
    help='Gpu rank')

parser.add_argument(
    '--resume', type=bool, default=False, # True False
    help='Resuming from existing model')

parser.add_argument(
    '--net', type=str, default='mdgat', 
    help='Choose net structure : mdgat superglue')

parser.add_argument(
    '--k', type=int, default=[128, None, 128, None, 64, None, 64, None], 
    help='Mdgat structure. None means connect all the nodes.')

parser.add_argument(
    '--l', type=int, default=9, 
    help='Layers number of GNN')

parser.add_argument(
    '--descriptor', type=str, default='FPFH', 
    help='Choose keypoint descriptor : FPFH pointnet pointnetmsg FPFH_gloabal FPFH_only')

parser.add_argument(
    '--keypoints', type=str, default='USIP', 
    help='Choose keypoints : sharp USIP lessharp')

parser.add_argument(
    '--ensure_kpts_num', type=bool, default=True, 
    help='')

parser.add_argument(
    '--max_keypoints', type=int, default=2048,  #1024
    help='Maximum number of keypoints'
            ' (\'-1\' keeps all keypoints)')

parser.add_argument(
    '--dataset', type=str, default='kitti',  
    help='Used dataset')

parser.add_argument(
    '--resume_model', type=str, default='./your_model.pth',
    help='Path to the resumed model')

parser.add_argument(
    '--train_path', type=str, default='./KITTI/', 
    help='Path to the directory of training scans.')

parser.add_argument(
    '--keypoints_path', type=str, default='./KITTI/keypoints/tsf_256_FPFH_16384-512-k1k16-2d-nonoise',
    help='Path to the directory of kepoints.')

parser.add_argument(
    '--txt_path', type=str, default='./KITTI/preprocess-random-full', 
    help='Path to the directory of pairs.')

parser.add_argument(
    '--model_out_path', type=str, default='./checkpoint',
    help='Path to the directory of output model')

parser.add_argument(
    '--match_threshold', type=float, default=0.1,
    help='SuperGlue match threshold')

parser.add_argument(
    '--threshold', type=float, default=0.5,
    help='Ground truth distance threshold')


def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title=''):
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    out = np.stack([out]*3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    Ht = int(H * 30 / 480)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (10, Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    H*1.0/480, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (10, Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    H*1.0/480, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out

        
if __name__ == '__main__':
    opt = parser.parse_args()
    
    from load_data import SparseDataset
    
    if opt.net == 'raw':
        opt.k = None
        opt.l = 9
    if opt.mutual_check:
        model_name = '{}-k{}-batch{}-{}-{}-{}' .format(opt.net, opt.k, opt.batch_size, opt.loss_method, opt.descriptor, opt.keypoints)
    else:
        model_name = 'nomutualcheck-{}-k{}-batch{}-{}-{}-{}' .format(opt.net, opt.k, opt.batch_size, opt.loss_method, opt.descriptor, opt.keypoints)
    
    log_path = './logs/{}/{}{}-k{}-{}-{}' .format(opt.dataset, opt.net, opt.l, opt.k, opt.loss_method, opt.descriptor)
    if opt.descriptor == 'pointnet' or opt.descriptor == 'pointnetmsg':
        log_path = '{}/train_step{}' .format(log_path, opt.train_step)
    log_path = '{}/{}' .format(log_path,model_name)
    log_path = Path(log_path)
    log_path.mkdir(exist_ok=True, parents=True)
    logger = SummaryWriter(log_path)
    
    model_out_path = '/workspace/MDGAT-matcher/outputs_gpu2'
    model_out_path = Path(model_out_path)
    model_out_path.mkdir(exist_ok=True, parents=True)

    print("Train",opt.net,"net with \nStructure k:",opt.k,"\nDescriptor: ",opt.descriptor,"\nLoss: ",opt.loss_method,"\nin Dataset: ",opt.dataset,
    "\n====================",
    "\nmodel_out_path: ", model_out_path,
    "\nlog_path: ",log_path)
   
    if opt.resume:        
        path_checkpoint = opt.resume_model  
        checkpoint = torch.load(path_checkpoint) 
        lr = checkpoint['lr_schedule']  # lr = opt.learning_rate # lr = checkpoint['lr_schedule']
        start_epoch = 1  # start_epoch = 1 # start_epoch = checkpoint['epoch'] + 1 
        loss = checkpoint['loss']
        best_loss = 1
    else:
        start_epoch = 1
        best_loss = 1e6
        lr=opt.learning_rate
    
    config = {
            'net': {
                'sinkhorn_iterations': opt.sinkhorn_iterations,
                'match_threshold': opt.match_threshold,
                'lr': lr,
                'k': opt.k,
                'descriptor': opt.descriptor,
                'L':opt.l
            }
        }
    
    net = MDGAT(config.get('net', {}))
    
    if torch.cuda.is_available():
        net.cuda() # make sure it trains on GPU
    else:
        print("### CUDA not available ###")

    if opt.resume:
        net.load_state_dict(checkpoint['net']) 
        optimizer = torch.optim.Adam(net.parameters(), lr=config.get('net', {}).get('lr'), weight_decay=1e-4)
        print('Resume from:', opt.resume_model, 'at epoch', start_epoch, ',loss', loss, ',lr', lr,'.\nSo far best loss',best_loss,
        "\n====================")
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
        print('====================\nStart new training')


    # load training data
    train_set = SparseDataset('/workspace/yfcc100m', 'train', opt.max_keypoints, 'superpoint')
    train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=False, batch_size=opt.batch_size, drop_last=True)
    print(f'Training data loaded with {len(train_set)} images.')

    # load validation data
    valid_set = SparseDataset('/workspace/yfcc100m', 'val', opt.max_keypoints, 'superpoint')
    valid_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=False, batch_size=opt.batch_size, drop_last=True)
    print(f'Validation data loaded with {len(valid_set)} images.')
    
    mean_loss = []
    for epoch in range(start_epoch, opt.epoch+1):
        epoch_loss = 0
        net.float().train()

        begin = time.time()
        for i, pred in enumerate(tqdm(train_loader)):
            for k in pred:
                if k != 'file_name' and k!='image0' and k!='image1':
                    if type(pred[k]) == torch.Tensor:
                        pred[k] = Variable(pred[k].cuda())
                    else:
                        pred[k] = Variable(torch.stack(pred[k]).cuda())
            
            data = net(pred)

            for k, v in pred.items():
                if k != 'all_matches':
                    pred[k] = v[0]
            pred = {**pred, **data}

            if pred['skip_train'] == True: # image has no keypoint
                continue

            optimizer.zero_grad()
            Loss = net.loss(pred)
            epoch_loss += Loss.item()
            mean_loss.append(Loss)
            Loss.backward()
            optimizer.step()
            # lr_schedule.step()

            # for every 1000 images, print progress and visualize the matches
            if (i+1) % 1000 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}' 
                    .format(epoch, opt.epoch, i+1, len(train_loader), torch.mean(torch.stack(mean_loss)).item()))
                mean_loss = []

                ### eval ###
                # Visualize the matches.
                image0, image1 = pred['image0'].cpu().numpy()[0]*255., pred['image1'].cpu().numpy()[0]*255.
                kpts0, kpts1 = pred['keypoints0'].cpu().numpy()[0], pred['keypoints1'].cpu().numpy()[0]
                matches, conf = pred['matches0'].cpu().detach().numpy()[0], pred['matching_scores0'].cpu().detach().numpy()[0]
                valid = matches > -1
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                mconf = conf[valid]
                viz_path = '{}/{}_matches.png'.format(model_out_path, str(i+1))
                color = cm.jet(mconf)
                stem = pred['file_name']
                text = [
                    'Training',
                    'Keypoints: %d x %d' % (len(kpts0), len(kpts1)),
                    'Matches: %d' % len(mkpts0),
                ]
                make_matching_plot_fast(
                    image0, image1, kpts0, kpts1, mkpts0, mkpts1, color, text,
                    path=viz_path, opencv_title='Matches')

            del Loss, pred, data, i

        # validation
        '''
            model.eval():   will notify all your layers that you are in eval mode, 
                            that way, batchnorm or dropout layers will work in eval 
                            mode instead of training mode.
            torch.no_grad():impacts the autograd engine and deactivate it. It will 
                            reduce memory usage and speed up computations but you 
                            won’t be able to backprop (which you don’t want in an eval script).
        '''
        net.eval()
        with torch.no_grad():
            mean_val_loss = 0
            for i, pred in enumerate(valid_loader):
                if i == 100:
                    break
                for k in pred:
                    if k != 'file_name' and k!='image0' and k!='image1':
                        if type(pred[k]) == torch.Tensor:
                            pred[k] = Variable(pred[k].cuda())
                        else:
                            pred[k] = Variable(torch.stack(pred[k]).cuda())
                
                data = net(pred)
                for k, v in pred.items(): 
                    if k != 'all_matches':
                        pred[k] = v[0]
                pred = {**pred, **data}

                if pred['skip_train'] == True: # image has no keypoint
                    continue

                Loss = net.loss(pred)
                mean_val_loss += Loss.item()

                if i < 10:
                    image0, image1 = pred['image0'].cpu().numpy()[0]*255., pred['image1'].cpu().numpy()[0]*255.
                    kpts0, kpts1 = pred['keypoints0'].cpu().numpy()[0], pred['keypoints1'].cpu().numpy()[0]
                    matches, conf = pred['matches0'].cpu().detach().numpy()[0], pred['matching_scores0'].cpu().detach().numpy()[0]
                    valid = matches > -1
                    mkpts0 = kpts0[valid]
                    mkpts1 = kpts1[matches[valid]]
                    mconf = conf[valid]
                    viz_path = '{}/Valid_{}_matches.png'.format(model_out_path, str(i+1))
                    color = cm.jet(mconf)
                    stem = pred['file_name']
                    text = [
                        'Validation',
                        'Keypoints: %d x %d' % (len(kpts0), len(kpts1)),
                        'Matches: %d' % len(mkpts0),
                    ]

                    make_matching_plot_fast(
                        image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                        text, viz_path)
         
            mean_val_loss /= len(valid_loader)
            epoch_loss /= len(train_loader)
            print('Validation loss: {:.4f}, epoch_loss: {:.4f},  best val loss: {:.4f}' .format(mean_val_loss, epoch_loss, best_loss))
            checkpoint = {
                    "net": net.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    "epoch": epoch,
                    'lr_schedule': optimizer.state_dict()['param_groups'][0]['lr'],
                    'loss': mean_val_loss
                }
            if (mean_val_loss <= best_loss + 1e-5): 
                best_loss = mean_val_loss
                model_out_fullpath = "{}/best_model_epoch_{}(val_loss{}).pth".format(model_out_path, epoch, best_loss)
                torch.save(checkpoint, model_out_fullpath)
                print('So far best loss: {:.4f}, Checkpoint saved to {}' .format(best_loss, model_out_fullpath))
            else:
                model_out_fullpath = "{}/model_epoch_{}.pth".format(model_out_path, epoch)
                torch.save(checkpoint, model_out_fullpath)
                print("Epoch [{}/{}] done. Epoch Loss {:.4f}. Checkpoint saved to {}"
                    .format(epoch, opt.epoch, epoch_loss, model_out_fullpath))

            # ================================================================== #
            #                        Tensorboard Logging                         #
            # ================================================================== #
            logger.add_scalar('Train/val_loss',mean_val_loss,epoch)
            logger.add_scalar('Train/epoch_loss',epoch_loss,epoch)
            print("log file saved to {}\n"
                .format(log_path))