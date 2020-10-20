import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

import triple_transforms
from nets import depth_predciton, basic, basic_NL, DGNLNet
from config import train_raincityscapes_path, test_raincityscapes_path
from dataset3 import ImageFolder
from misc import AvgMeter, check_mkdir

# torch.cuda.set_device(0)

cudnn.benchmark = True

ckpt_path = './ckpt'
exp_name = 'basic'

args = {
    'iter_num': 40000,
    'train_batch_size': 4,
    'last_iter': 0,
    'lr': 5e-4,
    'lr_decay': 0.9,
    'weight_decay': 0,
    'momentum': 0.9,
    'resume_snapshot': '',
    'val_freq': 50000000,
    'img_size_h': 512,
	'img_size_w': 1024,
	'crop_size': 512,
    'snapshot_epochs': 10000
}

transform = transforms.Compose([
    transforms.ToTensor()
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_pil = transforms.ToPILImage()

triple_transform = triple_transforms.Compose([
    triple_transforms.Resize((args['img_size_h'], args['img_size_w'])),
    #triple_transforms.RandomCrop(args['crop_size']),
    triple_transforms.RandomHorizontallyFlip()
])

train_set = ImageFolder(train_raincityscapes_path, transform=transform, target_transform=transform, triple_transform=triple_transform, is_train=True)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=8, shuffle=True)
test1_set = ImageFolder(test_raincityscapes_path, transform=transform, target_transform=transform, is_train=False)
test1_loader = DataLoader(test1_set, batch_size=2)


criterion = nn.L1Loss()
criterion_depth = nn.L1Loss()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')


def main():
    net = basic().cuda().train()

    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias' and param.requires_grad],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ])

    if len(args['resume_snapshot']) > 0:
        print('training resumes from \'%s\'' % args['resume_snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['resume_snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['resume_snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)


def train(net, optimizer):
    curr_iter = args['last_iter']
    while True:
        train_loss_record = AvgMeter()
        train_net_loss_record = AvgMeter()

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

            inputs, gts, dps = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            gts = Variable(gts).cuda()
            dps = Variable(dps).cuda()

            optimizer.zero_grad()

            result = net(inputs)

            loss_net = criterion(result, gts)


            loss = loss_net

            loss.backward()

            optimizer.step()

            # for n, p in net.named_parameters():
            #     if n[-5:] == 'alpha':
            #         print(p.grad.data)
            #         print(p.data)

            train_loss_record.update(loss.data, batch_size)
            train_net_loss_record.update(loss_net.data, batch_size)

            curr_iter += 1

            log = '[iter %d], [train loss %.5f], [lr %.13f], [loss_net %.5f]' % \
                  (curr_iter, train_loss_record.avg, optimizer.param_groups[1]['lr'],
                   train_net_loss_record.avg)
            print(log)
            open(log_path, 'a').write(log + '\n')

            if (curr_iter + 1) % args['val_freq'] == 0:
                validate(net, curr_iter, optimizer)

            if (curr_iter + 1) % args['snapshot_epochs'] == 0:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, ('%d.pth' % (curr_iter + 1) )))
                torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, ('%d_optim.pth' % (curr_iter + 1) )))

            if curr_iter > args['iter_num']:
                return


def validate(net, curr_iter, optimizer):
    print('validating...')
    net.eval()

    loss_record1, loss_record2 = AvgMeter(), AvgMeter()
    iter_num1 = len(test1_loader)

    with torch.no_grad():
        for i, data in enumerate(test1_loader):
            inputs, gts, dps = data
            inputs = Variable(inputs).cuda()
            gts = Variable(gts).cuda()
            dps = Variable(dps).cuda()

            res = net(inputs)

            loss = criterion(res, gts)
            loss_record1.update(loss.data, inputs.size(0))

            print('processed test1 %d / %d' % (i + 1, iter_num1))


    snapshot_name = 'iter_%d_loss1_%.5f_loss2_%.5f_lr_%.6f' % (curr_iter + 1, loss_record1.avg, loss_record2.avg,
                                                               optimizer.param_groups[1]['lr'])
    print('[validate]: [iter %d], [loss1 %.5f], [loss2 %.5f]' % (curr_iter + 1, loss_record1.avg, loss_record2.avg))
    torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '_optim.pth'))

    net.train()


if __name__ == '__main__':
    main()
