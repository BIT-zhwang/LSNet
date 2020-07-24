from cores.net import vgg16,lenet5
import torch.utils.data as Data
from torchvision import datasets,transforms
import torch.nn.functional as F
import torch
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train(args, model, device, train_loader, optimizer, epoch, logger):
    model=model.train()
    #criterion=torch.nn.CrossEntropyLoss()
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, label)
        loss=F.cross_entropy(output, label)
        #loss=criterion(output,label)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            # tensorboard
            logger.add_scalar("loss", loss.item(), batch_idx * len(data)+(epoch-1)*len(train_loader.dataset))

def test(args, model, device, test_loader, epoch, logger):
    model.eval()
    #criterion=torch.nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            #print(label)
            data, label = data.to(device), label.to(device)
            #print("data.shape",data.shape)
            output = model(data)
            #test_loss += criterion(output, label, size_average=False).item() # sum up batch loss
            test_loss += F.cross_entropy(output, label).item() 
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability [[0]]
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    if epoch!=0:
        logger.add_scalar("accu", correct / len(test_loader.dataset), epoch)
    return correct / len(test_loader.dataset)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num-classes', type=int, default=10, metavar='S',
                        help='the number of classes (default: 10)')
    parser.add_argument('--mode', type=int, default=1, metavar='S',
                        help='train or test (default: 1)')
    parser.add_argument('--checkpoints', type=int, default=2, metavar='S',
                        help='how many steps to store checkpoints (default: 1)')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device=torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_dataset=datasets.ImageFolder('./data/train_model/train',
                                          transform=transforms.Compose([
                                              transforms.Resize((32,32), interpolation=2),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
                                          )
    # dataset->dataloader
    train_loader = Data.DataLoader(
        dataset=train_dataset,  # torch TensorDataset format
        batch_size=args.batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        **kwargs
    )
    test_dataset=datasets.ImageFolder('./data/train_model/test',
                                          transform=transforms.Compose([
                                              transforms.Resize((32,32), interpolation=2),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
                                          )
    # dataset->dataloader
    test_loader = Data.DataLoader(
        dataset=test_dataset,  # torch TensorDataset format
        batch_size=1,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        **kwargs
    )
    model = lenet5().to(device)
    #model = vgg16().to(device)
    print("Network Structure:",model)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.8)
    #criterion = torch.nn.CrossEntropyLoss()
    
    logger=SummaryWriter()
    max_accu=0.0
    if args.mode==1:
        for epoch in range(1, args.epochs + 1):
            #scheduler.step()
            train(args, model, device, train_loader, optimizer, epoch, logger)
            accu=test(args, model, device, test_loader, epoch, logger)
            if accu>max_accu:
                max_accu=accu
                torch.save(model.state_dict(),'./weights/lenet5_best.pth')
            else:
                if epoch%args.checkpoints==0:
                    torch.save(model.state_dict(),'./checkpoints/lenet5_epoch'+str(epoch)+'.pth')

        #save model
        torch.save(model.state_dict(),'./weights/vgg16.pth')
    else:
        model.load_state_dict(torch.load('./weights/vgg16_best.pth'))
        test(args, model, device, test_loader, 0, logger)