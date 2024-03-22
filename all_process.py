import argparse, os, re, pickle
import numpy as np
from model import *
from dataset_3d import *
from utils import denorm, AverageMeter, ConfusionMeter, calc_loss, calc_accuracy, save_checkpoint, write_log

import torch
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--final_dim', default=1024, type=int, help='length of vector output from audio/video subnetwork')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--resume', default='true', type=str, help='path of model to resume training')
parser.add_argument('--start-epoch', default=12, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--img_dim', default=224, type=int)
parser.add_argument('--out_dir',default="/data/dfdc",type=str, help='Output directory containing Deepfake_data')
parser.add_argument('--print_freq', default=100, type=int, help='frequency of printing output during training')
parser.add_argument('--hyper_param', default=0.99, type=float, help='margin hyper parameter used in loss equation')
parser.add_argument('--threshold', default=0.3, type=float, help='threshold for testing')
parser.add_argument('--test', default='', type=str)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--log_dir',default='./log/dfdc.log',type=str, help='')
parser.add_argument('--tensorboard_dir',default="./tensorboard/dfdc",type=str, help='')
parser.add_argument('--checkpoints',default="./log_tmp/dfdc",type=str, help='')

def main():
	torch.manual_seed(0)
	np.random.seed(0)
	global args; args = parser.parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"]="1"
	global cuda; cuda = torch.device('cuda')
	if not os.path.exists(args.tensorboard_dir):
		os.makedirs(args.tensorboard_dir)
	if not os.path.exists(args.checkpoints):
		os.makedirs(args.checkpoints)
	global iteration; iteration = 0
	global iteration_val; iteration_val = 0
  
	num_classes=2
	two_stream_model = TwoStreamNetwork(num_classes)

	two_stream_model=two_stream_model.to(cuda)
	global criterion; criterion = nn.CrossEntropyLoss()

	params = two_stream_model.parameters()
	optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)

	transform = transforms.Compose([
		Scale(size=(args.img_dim,args.img_dim)),
		ToTensor(),
		Normalize()
	])
	if args.resume:
		checkpoint = torch.load('/checkpoints/model.pth')
		two_stream_model.load_state_dict(checkpoint)
		print("load model sucessed")


	if args.test:
		if os.path.isfile(args.test):
			print("=> loading testing checkpoint '{}'".format(args.test))
			two_stream_model.load_state_dict(torch.load(args.test))

		transform = transforms.Compose([
		Scale(size=(args.img_dim,args.img_dim)),
		ToTensor(),
		Normalize()
		])
		test_loader = get_data(transform, 'test')
		loss_avg,val_accuracy,auc = test(test_loader, two_stream_model)
		sys.exit()
  		
	else: # not test
		torch.backends.cudnn.benchmark = True


	train_loader,val_loader= get_data(transform, 'train')
	model_acc=0.0	
	for epoch in range(args.start_epoch, args.epochs):
		train_acc,val_acc,train_loss,val_loss=train(train_loader,val_loader,two_stream_model,optimizer,criterion,epoch,args.tensorboard_dir)

		f=f'{epoch}_{val_acc}.pth'
		file_name = os.path.join(args.checkpoints,f)
		torch.save(two_stream_model.state_dict(), file_name)
			# model_acc=val_acc
		log_message = f'Epoch {epoch}, train_acc: {train_acc:.4f}, train_loss: {train_loss:.4f}, val_acc: {val_acc:.4f}, val_loss: {val_loss:.4f}\n'
		with open(args.log_dir, 'a') as file:
			file.write(log_message)


def get_data(transform, mode=''):
    print('Loading data for "%s" ...' % mode)
    if mode == 'train':
        split_file = os.path.join(args.out_dir, 'train_split.csv')
        video_info = pd.read_csv(split_file, header=None)
        train_video_info, val_video_info = train_test_split(video_info, test_size=0.1, random_state=42)
        
        train_data = deepfake_data(train_video_info,mode=mode,transform=transform)
        val_data = deepfake_data(val_video_info,mode=mode,transform=transform)
        
        sampler1 = data.RandomSampler(train_data)
        sampler2 = data.RandomSampler(val_data) 
        
        train_loader = data.DataLoader(train_data,
                                      batch_size=args.batch_size,
                                      sampler=sampler1,
                                      shuffle=False,
                                      num_workers=4,
                                      pin_memory=True,
                                      drop_last=True,
                                      collate_fn=my_collate)
        
        val_loader = data.DataLoader(val_data,
                                      batch_size=args.batch_size,
                                      sampler=sampler2,
                                      shuffle=False,
                                      num_workers=4,
                                      pin_memory=True,
                                      drop_last=True,
                                      collate_fn=my_collate)
        print('"%s" train dataset size: %d;val dataset size: %d' % (mode, len(train_data),len(val_data)))
        return train_loader,val_loader
        
    elif mode == 'test':
        dataset = deepfake_data(args.out_dir,mode=mode,
                         transform=transform)
    
        sampler = data.RandomSampler(dataset)
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=8,
                                      pin_memory=True,
                                      collate_fn=my_collate)
        print('"%s" dataset size: %d' % (mode, len(dataset)))
        return data_loader

def my_collate(batch):
	batch = list(filter(lambda x: x is not None, batch))
	return torch.utils.data.dataloader.default_collate(batch)


def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0) # 32
    # get top-k index
    _, pred = output.topk(maxk, 1, True, True) # using top-k to get the index of the top k
    pred = pred.t() # transpose
    # eq: compare according to corresponding elements; view(1, -1): automatically converted to the shape of row 1, ;
    # expand_as(pred): shape extended to 'pred'
    # expand_as performs row-by-row replication to expand, and ensure that the columns are equal
    correct = pred.eq(label.view(1, -1).expand_as(pred))
    # print(correct)

    rtn = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0) # flatten the data of first k row to one dim to get total numbers of true
        rtn.append(correct_k.mul_(100.0 / batch_size)) # mul_() tensor's multiplication, (acc num/total num)*100 to become percentage
    return rtn
class AverageMeter_acc(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        
def train(train_loader,val_loader,two_stream_model,optimizer,criterion,epoch,tensorboard_path):
	two_stream_model.train()
	top1 = AverageMeter_acc() # metric
	train_loss = 0.0
	numb=0
	writer = SummaryWriter(tensorboard_path)
	for idx, (t_seq, cct, target, audiopath) in enumerate(train_loader):
		tic = time.time()
		#coming soon
		# print(loss.item())
		train_loss += loss.item()
		numb +=1
		global iteration
		print('Train Epoch: {} [{}/{} ]\t Loss: {:.6f}  \t train_acc: {:.2f} \t train_avg_acc: {:.6f}'
				.format(epoch,idx,len(train_loader), loss.item(),acc1,top1.avg))
		if idx % args.print_freq == 0:
			writer.add_scalar('local/loss',train_loss / numb, iteration)
			iteration+=1
		if idx % 100 == 0:
			f=f'{epoch}_{idx}_{top1.avg}.pth'
			file_name = os.path.join(args.checkpoints,f)
			torch.save(two_stream_model.state_dict(), file_name)
			log_message = f'Epoch {epoch}, Idx {idx}, train_acc: {top1.avg:.4f}, train_loss: {train_loss / numb:.4f}\n'
			with open(args.log_dir, 'a') as file:
				file.write(log_message)

	writer.add_scalar('Train/Loss',  train_loss / numb, epoch)
	writer.add_scalar('Train/Accuracy', top1.avg, epoch)

	loss_avg = train_loss / numb
	print('Train Epoch: {}  avg_loss : {:.6f} \t train_acc: {:.6f}'.format(epoch, loss_avg,top1.avg))
    
	two_stream_model.eval()
	with torch.no_grad():
		val_top1 = AverageMeter_acc()
		val_loss = 0.0
		total=0
		for idx, (t_seq, cct, target, audiopath) in enumerate(val_loader):
			tic = time.time()
			t_seq, cct, target = t_seq.to(cuda), cct.to(cuda), target.to(cuda)
			(B,P,C,H,W)=t_seq.size()
			reshaped_seq = torch.reshape(t_seq, (B, C, H*P, W))
			cct = cct.unsqueeze(1).type(torch.double)
			out = two_stream_model(reshaped_seq,cct)
			# loss = criterion(out,torch.eye(2, device=cuda)[target].view((len(target)), -1))
			loss = criterion(out,target.view(-1))

			val_loss += loss.item()
			acc1, acc2 = accuracy(out, target, topk=(1,2))

			val_top1.update(acc1.item(), B)

			total += 1
			global iteration_val

			if idx % args.print_freq == 0:
				writer.add_scalar('local_val/loss',val_loss / total, iteration_val)
				iteration_val += 1
			if idx % 100 == 0:
				log_message = f'Epoch {epoch}, Idx {idx}, val_acc: {val_top1.avg:.4f}, val_loss: {val_loss / total:.4f}\n'
				with open(args.log_dir, 'a') as file:
					file.write(log_message)

			# correct += (predicted == target).sum().item()
		avg_val_loss = val_loss / total
		val_accuracy = val_top1.avg
		writer.add_scalar('Val/Loss', avg_val_loss, epoch)
		writer.add_scalar('Val/Accuracy', val_top1.avg, epoch)
		writer.flush()   
	print(f'Epoch {epoch+1}, validation_loss: {avg_val_loss:.4f}, validation_acc: {val_accuracy:.4f}%')
	return top1.avg,val_accuracy,loss_avg,avg_val_loss
 
def softmax(z):
    exp_z = np.exp(z - np.max(z))  #Subtract the maximum value to avoid exponential overflow
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def test(data_loader, model):
	model.eval()
	tar=[]
	pre=[]
	with torch.no_grad():
		#coming soon
	roc_auc = roc_auc_score(tar, pre)

	print(f'ROC AUC: {roc_auc:.4f}')
	print(f'test_loss: {avg_val_loss:.4f}, test_acc: {val_accuracy:.4f}, test_auc: {roc_auc:.4f}%')
	return avg_val_loss,val_accuracy,roc_auc



if __name__ == '__main__':
    main()

