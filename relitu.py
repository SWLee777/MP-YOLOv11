import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, cv2, os, shutil
import numpy as np

np.random.seed(0)
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import DetectionModel as Model
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy

from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    XGradCAM
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import ActivationsAndGradients


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    将图像进行 resize + padding（同 YOLOv5 的 letterbox 操作）
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 只缩小，不放大
        r = min(r, 1.0)

    ratio = (r, r)  # width, height ratios
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # 缩放后的 w,h
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 需要的 padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    dw /= 2  # 将填充均分到两边
    dh /= 2

    # resize
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    # 填充
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


class yolov11_heatmap:
    def __init__(self, weight, cfg, device, method, layer, backward_type, conf_threshold, ratio):
        """
        :param weight:    训练好的权重
        :param cfg:       对应的模型yaml
        :param device:    'cpu' or 'cuda:0'
        :param method:    'GradCAM', 'GradCAMPlusPlus', 'XGradCAM'
        :param layer:     选择哪一层做可视化
        :param backward_type: 'class', 'box', or 'all'
        :param conf_threshold: 置信度阈值
        :param ratio:     用于截断排序后检测框数量的一个比例
        """
        device = torch.device(device)
        ckpt = torch.load(weight, map_location=device)
        model_names = ckpt['model'].names
        csd = ckpt['model'].float().state_dict()

        model = Model(cfg, ch=3, nc=len(model_names)).to(device)
        csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])  # 过滤不匹配的层
        model.load_state_dict(csd, strict=False)
        model.eval()
        print(f'Transferred {len(csd)}/{len(model.state_dict())} items')

        # pytorch_grad_cam 需要的参数
        target_layers = [eval(layer)]  # layer='model.model[9]' 例如
        method = eval(method)          # method='GradCAM' / 'GradCAMPlusPlus' / 'XGradCAM'

        # 生成随机颜色
        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int32)

        # 把所有局部变量挂到 self 上
        self.__dict__.update(locals())

    def post_process(self, result):
        """
        对模型输出进行后处理：
        result 的形状是 [1, num_boxes, 4 + num_classes]
        返回:
          post_result:    shape=[num_boxes, num_classes] 的置信度
          pre_post_boxes: shape=[num_boxes, 4] 原 xywh
          post_boxes:     shape=[num_boxes, 4] 转成 xyxy 后的坐标
        """
        if result.size(0) == 0:
            return None, None, None

        # 取出 logits_ = [1, num_boxes, num_classes]
        # boxes_  = [1, num_boxes, 4]
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]

        # 对每个框取最大置信度
        # sorted, indices 返回: 最大置信度值 的 排序
        sorted_scores, indices = torch.sort(logits_.max(1)[0], descending=True)

        # 取置信度对应的索引，然后把 logits_ 与 boxes_ 都根据 indices[0] 排序
        # logits_[0] shape=[num_boxes, num_classes]； 取转置再根据 indices[0] 排序
        sorted_logits = torch.transpose(logits_[0], 0, 1)[indices[0]]
        sorted_xywh   = torch.transpose(boxes_[0], 0, 1)[indices[0]]
        sorted_xyxy   = xywh2xyxy(sorted_xywh).cpu().detach().numpy()

        return sorted_logits, sorted_xywh, sorted_xyxy

    def draw_detections(self, box, color, name, img):
        """
        在图上画框 + 类别名称
        """
        xmin, ymin, xmax, ymax = list(map(int, box))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    tuple(int(x) for x in color), 2,
                    lineType=cv2.LINE_AA)
        return img

    def __call__(self, img_path, save_path):
        # 如果已存在同名文件夹，删除
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

        # 读取并预处理图像
        img = cv2.imread(img_path)
        img = letterbox(img)[0]  # resize + padding
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0

        # 转成 tensor
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        # 初始化 ActivationsAndGradients
        grads = ActivationsAndGradients(self.model, self.target_layers, reshape_transform=None)

        # 前向推理
        result = grads(tensor)

        if result is None or len(result) == 0 or result[0].size(0) == 0:
            print("No valid detections found.")
            return

        # 后处理
        post_result, pre_post_boxes, post_boxes = self.post_process(result[0])
        if post_result is None:
            print("No valid post_result.")
            return

        # activations: 某个中间层的输出
        activations = grads.activations[0].cpu().detach().numpy()

        # 遍历置信度从大到小的检测框
        num_det = int(post_result.size(0) * self.ratio)  # 只取前 ratio 百分比的框
        for i in trange(num_det):
            # 如果这个框的最大置信度都 < conf_threshold，直接结束
            if float(post_result[i].max()) < self.conf_threshold:
                break

            # 先清空梯度
            self.model.zero_grad()

            # 根据 backward_type 来决定回传哪个分量的梯度
            # 先对 "类别" 分量回传梯度
            if self.backward_type == 'class' or self.backward_type == 'all':
                score = post_result[i].max()  # 取该检测框最大的置信度
                score.backward(retain_graph=True)

            # 再对 "box" 分量回传梯度
            if self.backward_type == 'box' or self.backward_type == 'all':
                # pre_post_boxes[i] 是 [x,y,w,h]
                for j in range(4):
                    score_box = pre_post_boxes[i, j]
                    score_box.backward(retain_graph=True)

            # 获取梯度
            # grads.gradients 里会有多个，因为我们对 class 和 box 都 backward 了
            # 在官方示例中，不同方法对梯度的合并方式不一样。这里按照原作者的写法来
            if self.backward_type == 'class':
                gradients = grads.gradients[0]
            elif self.backward_type == 'box':
                # box 4个通道合并
                # grads.gradients[0..3] 可能分别是每个box分量
                # 这里需要看你实际梯度存储结构，如果只存了一次，需要自己调试
                gradients = sum(grads.gradients[:4])
            else:
                # class + box 都加起来
                # grads.gradients 里依次可能是 class, box_x, box_y, box_w, box_h
                # 具体情况看你上面 backward 的顺序
                gradients = sum(grads.gradients)

            # 计算 CAM 的权重
            b, k, u, v = gradients.size()
            # pytorch_grad_cam 的 get_cam_weights 接口需要 activations 和 numpy 格式的 gradients
            weights = self.method.get_cam_weights(
                self.method,
                None,  # target_category
                None,  # activations
                None,  # grads
                activations,
                gradients.detach().numpy()
            )
            weights = weights.reshape((b, k, 1, 1))

            # 计算 saliency_map
            saliency_map = np.sum(weights * activations, axis=1)  # [b, 1, h, w] => [b, h, w]
            saliency_map = np.squeeze(np.maximum(saliency_map, 0))  # ReLU
            saliency_map = cv2.resize(saliency_map, (tensor.size(3), tensor.size(2)))
            # 归一化到 [0,1]
            sm_min, sm_max = saliency_map.min(), saliency_map.max()
            if (sm_max - sm_min) < 1e-7:
                # 如果全是同一个值，就跳过
                continue
            saliency_map = (saliency_map - sm_min) / (sm_max - sm_min)

            # 将热力图叠加到原图上
            cam_image = show_cam_on_image(img.copy(), saliency_map, use_rgb=True)

            # 画上检测框和置信度
            color_idx = int(post_result[i, :].argmax())
            # 避免越界
            color_idx = min(color_idx, len(self.colors) - 1)
            cls_name = self.model_names[color_idx]
            cls_conf = float(post_result[i].max())

            ''''cam_image = self.draw_detections(
                post_boxes[i],
                self.colors[color_idx],
                f'{cls_name} {cls_conf:.2f}',
                cam_image
            )'''

            # 保存结果图
            cam_image = Image.fromarray(cam_image)
            cam_image.save(os.path.join(save_path, f'{i}.png'))


def get_params():
    params = {
        'weight': r'root_path\weights\best.pt',
        'cfg': r'root_path\ultralytics\cfg\models\11\yolo11-obb-MP.yaml',
        'device': 'cuda:0',
        'method': 'GradCAM',      # 可选：GradCAM, GradCAMPlusPlus, XGradCAM
        'layer': 'model.model[9]', # 你要做可视化的层
        'backward_type': 'box',   # 可选：'class' / 'box' / 'all'
        'conf_threshold': 0.01,   # 置信度阈值
        'ratio': 0.09             # 只取前 ratio 的检测框进行可视化
    }
    return params


if __name__ == '__main__':
    model = yolov11_heatmap(**get_params())
    model(r'root_path\testimg', 'result')
