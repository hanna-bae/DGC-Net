import torch 
import torch.nn as nn
import argparse
import onnx 
from model.net import DGCNet

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', metavar='ARCH', default='DGCNet')
    parser.add_argument('--ckpt', type=str, default='./model/dgcnet.pth')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    # Create the model and load the weights 
    model = DGCNet()
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(torch.load(args.ckpt, map_location=device)['state_dict'])

    # Create dummy input:
    dummy_input = (torch.rand(1, 3, 240, 240), torch.rand(1, 3, 240, 240))

    torch.onnx.export(model, 
                      dummy_input, 
                      args.model+'.onnx',
                      verbose=True)
    print('Successfully exported model')