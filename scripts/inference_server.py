from PIL import Image
from functools import partial
import numpy as np
import argparse
from tfserve import TFServeApp
from io import BytesIO

def encode(request_data, tensor_name):
    tempBuff = BytesIO()
    tempBuff.write(request_data)
    tempBuff.seek(0)
    img = Image.open(tempBuff)
    img = np.asarray(img) / 255.0
    return {tensor_name: img} # TODO: I should really be checking img to make sure it is valid.

def decode(outputs, tensor_name):
    p = outputs[tensor_name] # TODO: I should really check what the model returns before responding to REST call.
    index = np.argmax(p)
    return {"label": int(index), "probability": float(p[index] / np.sum(p))}

def parse_args():
    parser = argparse.ArgumentParser(description='Inference Server for MNIST')
    parser.add_argument('--model', dest='model', type=str, default='model/frozenmodel.pb')
    parser.add_argument('--input', dest='input', type=str, default='import/input_tensor:0')
    parser.add_argument('--output', dest='output', type=str, default='import/output/softmax:0')
    parser.add_argument('--host', dest='host', type=str, default='127.0.0.1')
    parser.add_argument('--port', dest='port', type=int, default=5000)
    return parser.parse_args()

def main():
    args = parse_args()
    bounded_encode = partial(encode, tensor_name=args.input)
    bounded_decode = partial(decode, tensor_name=args.output)
    app = TFServeApp(args.model, [args.input], [args.output], bounded_encode, bounded_decode)
    app.run(args.host, args.port, debug=True)

if __name__ == '__main__':
    main()
