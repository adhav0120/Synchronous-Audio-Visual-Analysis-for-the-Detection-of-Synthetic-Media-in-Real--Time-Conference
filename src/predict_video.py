import torch
import torch.nn as nn
from torchvision import transforms
import cv2
from mtcnn.mtcnn import MTCNN
import os
import numpy as np
from collections import OrderedDict
import re
import math
from torch.utils import model_zoo

# =====================================================================================
# == EFFICIENTNET-PYTORCH MODEL DEFINITION (FINAL)                                 ==
# =====================================================================================
# This is the full, self-contained model architecture that perfectly matches your 
# saved .pth file, based on the popular 'efficientnet-pytorch' implementation.

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Swish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class MBConvBlock(nn.Module):
    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip

        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            self._expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = nn.Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,
            kernel_size=k, stride=s, padding=(k - 1) // 2, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = nn.Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        final_oup = self._block_args.output_filters
        self._project_conv = nn.Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = Swish()

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(x)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        if self.has_se:
            x_squeezed = nn.functional.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x
        
        x = self._bn2(self._project_conv(x))

        if self.id_skip:
            if self._block_args.stride == 1 and self._block_args.input_filters == self._block_args.output_filters:
                if drop_connect_rate:
                    x = self.drop_connect(x, p=drop_connect_rate, training=self.training)
                x = x + inputs
        return x

    def drop_connect(self, inputs, p, training):
        if not training: return inputs
        batch_size = inputs.shape[0]
        keep_prob = 1 - p
        random_tensor = keep_prob
        random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
        binary_tensor = torch.floor(random_tensor)
        output = inputs / keep_prob * binary_tensor
        return output

class EfficientNet(nn.Module):
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        self._global_params = global_params
        self._blocks_args = blocks_args

        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon
        in_channels = 3
        out_channels = round_filters(32, self._global_params)
        self._conv_stem = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        in_channels = block_args.output_filters
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = Swish()

    def forward(self, inputs):
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        
        x = self._swish(self._bn1(self._conv_head(x)))
        x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc(x)
        return x

# Helper functions and classes for loading
def round_filters(filters, global_params):
    multiplier = global_params.width_coefficient
    if not multiplier: return filters
    divisor = global_params.depth_divisor
    min_depth = None or divisor
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, global_params):
    multiplier = global_params.depth_coefficient
    if not multiplier: return repeats
    return int(math.ceil(multiplier * repeats))

class BlockArgs(object):
    def __init__(self, num_repeat=None, kernel_size=None, stride=None, expand_ratio=None, input_filters=None,
                 output_filters=None, se_ratio=None, id_skip=None):
        self.num_repeat = num_repeat
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.se_ratio = se_ratio
        self.id_skip = id_skip

    def _replace(self, **kwargs):
        return BlockArgs(**{**self.__dict__, **kwargs})

class GlobalParams(object):
    def __init__(self, width_coefficient=None, depth_coefficient=None, image_size=None,
                 dropout_rate=None, num_classes=None, batch_norm_momentum=None,
                 batch_norm_epsilon=None, drop_connect_rate=None, depth_divisor=None,
                 min_depth=None, include_top=None):
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.image_size = image_size
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_epsilon = batch_norm_epsilon
        self.drop_connect_rate = drop_connect_rate
        self.depth_divisor = depth_divisor
        self.min_depth = min_depth
        self.include_top = include_top

# Default parameters for EfficientNet-B0
def efficientnet_params(model_name):
    params_dict = {
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    }
    w, d, s, p = params_dict[model_name]
    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    decoder = BlockDecoder()
    return decoder.decode(blocks_args), GlobalParams(
        width_coefficient=w, depth_coefficient=d, image_size=s, dropout_rate=p,
        num_classes=2, batch_norm_momentum=0.99, batch_norm_epsilon=1e-3,
        drop_connect_rate=0.2, depth_divisor=8, min_depth=None, include_top=True)

class BlockDecoder(object):
    def decode(self, string_list):
        return [self._decode_block_string(s) for s in string_list]

    def _decode_block_string(self, block_string):
        assert isinstance(block_string, str)
        parts = block_string.split('_')
        options = {}
        for part in parts:
            match = re.match(r'([a-z]+)([.\d]+)', part)
            if match:
                key, value = match.groups()
                options[key] = value
        
        stride = int(options['s'][0])

        return BlockArgs(
            num_repeat=int(options['r']),
            kernel_size=int(options['k']),
            stride=stride,
            expand_ratio=int(options['e']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            se_ratio=float(options['se']),
            id_skip=True)

# =====================================================================================
# == MODEL LOADING (FINAL)                                                         ==
# =====================================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

blocks_args, global_params = efficientnet_params('efficientnet-b0')
model = EfficientNet(blocks_args, global_params)
model = model.to(device)

model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'video_deepfake_detector.pth')
print(f"Loading video model from: {model_path}")

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# =====================================================================================

face_detector = MTCNN()

preprocess_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_video_deepfake(video_path: str, frame_interval: int = 30) -> float:
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return 0.0
        
        frame_scores = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = face_detector.detect_faces(frame_rgb)
                
                if faces:
                    face = max(faces, key=lambda f: f['confidence'])
                    x, y, w, h = face['box']
                    x, y = max(x, 0), max(y, 0)
                    face_crop = frame_rgb[y : y + h, x : x + w]
                    
                    if face_crop.size > 0:
                        input_tensor = preprocess_transform(face_crop).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            output = model(input_tensor)
                            probs = torch.softmax(output, dim=1)
                            score = probs[0, 1].item()
                            frame_scores.append(score)
            
            frame_count += 1
            
    except Exception as e:
        print(f"Error processing video: {e}")
        return 0.0
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
    
    return np.mean(frame_scores) if frame_scores else 0.0
