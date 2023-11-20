import argparse
import onnx
# from onnx_coreml import convert 
import coremltools as ct 

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', metavar='ARCH', default='DGCNet')
    parser.add_argument('--ckpt', type=str, default='./model/dgcnet.pth')
    parser.add_argument('--input_name', type=str, default='1')
    parser.add_argument('--image_scale', type=float, default=1./255.)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    # Load the onnx model as a coreml model 
    onnx_model = onnx.load(args.ckpt)
    model = ct.converters.sklearn.convert(onnx_model)
    model.save(args.model+'.mlmodel')