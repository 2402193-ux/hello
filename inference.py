import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from networks.vision_transformer import DeSwinUNet
from datasets.loader import MISdataset, ColorMode, MISdataset_RealMode
from PIL import Image
import skimage.metrics
from skimage.metrics import structural_similarity as ssim
import time
import argparse
import pandas as pd
from torch.cuda.amp import autocast

def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        try:
            # 使用utf-8而非ascii
            utf8_args = [str(arg).encode('utf-8', 'replace').decode('utf-8') for arg in args]
            print(*utf8_args, **kwargs)
        except Exception:
            print("Encoding error in print statement")


if os.path.exists("D:/StereoVideo"):
    sys.path.append("D:/StereoVideo")
elif os.path.exists("D:/StereoVideo"):
    sys.path.append("D:/StereoVideo")

try:
    from gmflow.gmflow import GMFlow

    print("Successfully imported GMFlow module")
except ImportError:
    for potential_path in [
        "D:/StereoVideo",
        "D:\\StereoVideo",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../StereoVideo"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../gmflow"),
        "D:/StereoVideo/gmflow",
        "../StereoVideo",
        "../gmflow",
        "../../gmflow",
        "../../StereoVideo",
    ]:
        if os.path.exists(potential_path):
            sys.path.append(potential_path)
            print(f"Adding a path to a Python search path: {potential_path}")
            try:
                from gmflow.gmflow import GMFlow

                print(f"Success from {potential_path} Importing GMFlow Modules")
                break
            except ImportError:
                print(f"Failed to import from{potential_path} ")
                continue
    else:
        print("Error: GMFlow module could not be found, please make sure it is installed correctly or provide the correct path")
        print("Current Python search path:", sys.path)
        raise ImportError("Unable to import GMFlow module, please make sure it is installed or provide the correct path")



def setup_gmflow_model(gmflow_checkpoint="D:/StereoVideo/gmflow/checkpoints/gmflow_sintel-0c07dcb3.pth"):
    print("Start initialising the GMFlow model...")

    if not os.path.exists(gmflow_checkpoint):
        print(f"Warning: GMFlow model weights do not exist {gmflow_checkpoint}")
        base_name = os.path.basename(gmflow_checkpoint)
        for potential_dir in [
            "D:/StereoVideo/gmflow/checkpoints",
            "D://StereoVideo//gmflow//checkpoints",
            "./checkpoints",
            "../checkpoints",
            "../../checkpoints",
        ]:
            potential_path = os.path.join(potential_dir, base_name)
            if os.path.exists(potential_path):
                gmflow_checkpoint = potential_path
                print(f"Finding GMFlow model weights: {gmflow_checkpoint}")
                break
        else:
            print(f"Error: GMFlow model weights could not be found. {base_name}")
            raise FileNotFoundError(f"GMFlow model weights file could not be found")

    model_flow = GMFlow(
        feature_channels=128,
        num_scales=1,
        upsample_factor=8,
        num_head=1,
        attention_type='swin',
        ffn_dim_expansion=4,
        num_transformer_layers=6
    ).cuda()

    # 加载预训练权重
    print(f'Loading GMFlow pre-trained models: {gmflow_checkpoint}')
    try:
        checkpoint = torch.load(gmflow_checkpoint, map_location='cpu')
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model_flow.load_state_dict(weights)
        model_flow.eval()
        print("GMFlow model loaded successfully")
    except Exception as e:
        print(f"Failed to load GMFlow model: {str(e)}")
        raise

    return model_flow


def compute_optical_flow(model_flow, img1, img2):
    """使用GMFlow计算两帧之间的光流"""
    if len(img1.shape) == 5:
        img1 = img1.squeeze(1)
        img2 = img2.squeeze(1)

    # 归一化到[-1, 1]
    img1_norm = img1 * 2 - 1
    img2_norm = img2 * 2 - 1

    with torch.no_grad():
        results_dict = model_flow(
            img1_norm, img2_norm,
            attn_splits_list=[2],
            corr_radius_list=[-1],
            prop_radius_list=[-1],
            pred_bidir_flow=False
        )
        flow = results_dict['flow_preds'][-1].detach()

    return flow


def calculate_metrics(output, target):
    """计算评估指标 - 修正版"""
    # 确保数据类型一致
    output = output.astype(np.float64)
    target = target.astype(np.float64)

    # 确保值在[0,1]范围内
    output = np.clip(output, 0, 1)
    target = np.clip(target, 0, 1)

    # 计算MSE
    mse_value = np.mean((output - target) ** 2)

    if mse_value == 0:
        psnr_value = float('inf')
    else:
        psnr_value = 10 * np.log10(1.0 / mse_value)

    # 计算SSIM（逐通道计算后取平均）
    ssim_channels = []
    for i in range(target.shape[2]):
        ssim_value = skimage.metrics.structural_similarity(
            target[:, :, i],
            output[:, :, i],
            data_range=1.0,
            win_size=11,
            gaussian_weights=True,
            K1=0.01,
            K2=0.03
        )
        ssim_channels.append(ssim_value)

    ssim_value = np.mean(ssim_channels)

    return psnr_value, ssim_value, mse_value


def process_frame(frame):
    frame = (frame.clamp(-1, 1) * 0.5 + 0.5).cpu().numpy()

    # 如果是4通道，只取前3个通道(RGB)
    if frame.shape[0] == 4:
        frame = frame[:3]
    # 确保是3通道图像
    elif frame.shape[0] != 3:
        if len(frame.shape) == 3 and frame.shape[-1] == 3:
            return frame  # 如果已经是HWC格式，直接返回
        elif len(frame.shape) == 2:
            frame = np.stack([frame] * 3, axis=0)  # 单通道转三通道

    return frame


def save_image(img_array, save_path):
    img_array = (img_array * 255).astype(np.uint8)
    if img_array.shape[0] == 3:
        img_array = np.transpose(img_array, (1, 2, 0))
    img_array = img_array[..., ::-1]
    img = Image.fromarray(img_array)
    img.save(save_path)


def inference_with_adaptive_mode(model_path, config, test_dir, output_dir,
                                 mode="virtual", sequence_length=5, flow_dir=None,
                                 gmflow_checkpoint="D:/StereoVideo/gmflow/checkpoints/gmflow_sintel-0c07dcb3.pth"):
    """

    改进版推理函数 - 支持两种模式：
    1. virtual: 使用预计算的光流，每5帧独立处理
    2. real: 实时计算光流，连续处理视频段落，使用上一组的最后一帧结果作为下一组的参考帧
    """
    safe_print(f"\nusing {mode} inference")

    # 性能跟踪字典
    perf_stats = {
        'model_load': 0,
        'data_load': 0,
        'forward_pass': 0,
        'flow_computation': 0,
        'postprocessing': 0,
        'saving': 0,
        'total': 0,
        'frame_count': 0
    }

    # 评估指标存储
    metrics_data = {
        'sequence_id': [],
        'frame_id': [],
        'psnr': [],
        'ssim': [],
        'mse': []
    }

    total_start = time.time()

    os.makedirs(output_dir, exist_ok=True)
    desmoking_dir = os.path.join(output_dir, 'desmoking')
    mask_dir = os.path.join(output_dir, 'mask')
    metrics_dir = os.path.join(output_dir, 'metrics')
    flow_output_dir = os.path.join(output_dir, 'flow')

    for dir_path in [desmoking_dir, mask_dir, metrics_dir, flow_output_dir]:
        os.makedirs(dir_path, exist_ok=True)

    gt_path = os.path.join(test_dir, "test", "gt")
    if not os.path.exists(gt_path):
        safe_print(f"warning: {gt_path} not exist")
        gt_path = os.path.join(test_dir, "gt")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"NO GT founding")

    if flow_dir is None and mode == "virtual":
        flow_dir = os.path.join(test_dir, "flow_npy")
        if not os.path.exists(flow_dir):
            safe_print(f"warning: {flow_dir} does not exist")

    model_load_start = time.time()

    # 加载模型
    print("loading de-smoking model...")
    model = DeSwinUNet(config, img_size=256, num_frames=sequence_length).cuda()

    try:
        state_dict = torch.load(model_path, map_location='cuda')
        try:
            model.load_state_dict(state_dict)
            print("Model loaded successfully - exact match")
        except Exception as e:
            # 修改这里，显示不匹配的详细信息
            incompatible = model.load_state_dict(state_dict, strict=False)
            print(f"Model loaded successfully - not a strict match")
            print(f"Missing parameters: {incompatible.missing_keys}")
            print(f"Unexpected parameters: {incompatible.unexpected_keys}")
    except Exception as e:
        print(f"Model loading failure: {str(e)}")
        return

    model.eval()
    import types

    if args.disable_edge:
        print("no edge")
        for edge_enhancer in model.edge_enhancers:
            def identity_forward(self, x):
                return x

            edge_enhancer.forward = types.MethodType(identity_forward, edge_enhancer)

    if args.disable_temporal:
        print("no reference")
        for temp_module in model.optical_flow_modules:
            def modified_forward(self, x_frames, flows=None, frame_names=None):
                B, T, C, H, W = x_frames.shape
                return [x_frames[:, t] for t in range(T)]

            temp_module.forward = types.MethodType(modified_forward, temp_module)

    if args.disable_flow:
        print("no flow")
        flows = None

    if args.disable_attention:
        print("no attention")
        for fusion_module in model.branch_fusion_modules:
            def simple_fusion(self, edge_feat, temp_feat):
                return (edge_feat + temp_feat) / 2.0

            fusion_module.forward = types.MethodType(simple_fusion, fusion_module)

    gmflow_model = None
    if mode == "real":
        print("Loading the GMFlow model for real-time timestream calculations...")
        gmflow_model = setup_gmflow_model(gmflow_checkpoint)

    perf_stats['model load'] = time.time() - model_load_start
    data_load_start = time.time()

    if mode == "virtual":
        print("Using MISdataset for virtual mode")
        test_dataset = MISdataset(
            data_dir=test_dir,
            mode="test",
            transform=None,
            color_model=ColorMode.BGR,
            sequence_length=sequence_length,
            load_flow=True,
            flow_dir=flow_dir,
            use_gt_first_frame=True
        )
    else:  # mode == "real"
        print("Using MISdataset_RealMode for real mode")
        test_dataset = MISdataset_RealMode(
            data_dir=test_dir,
            mode="test",
            transform=None,
            color_model=ColorMode.BGR,
            sequence_length=sequence_length,
            stride=sequence_length - 1,
            load_flow=False,
            flow_dir=None,
            use_gt_first_frame=True
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    perf_stats['data load'] = time.time() - data_load_start

    print(f"found {len(test_dataset)} sequences")
    print("Starting inference")

    # 帧计数器
    frame_count = 0
    last_frame_smoke_density = 0.0

    def inference_with_adaptive_weights(model, source_frames, flows, frame_names, last_smoke_mask=None):
        """带自适应权重的推理"""

        # 第一阶段：获取烟雾掩码
        with torch.no_grad():
            _, smoke_masks = model(source_frames, flows=flows, frame_names=frame_names)

        if last_smoke_mask is not None:
            smoke_masks[:, 0] = last_smoke_mask

        for module in model.swin_unet.optical_flow_modules:
            module.set_smoke_masks(smoke_masks)
            module.set_adaptive_weights(True)

        with torch.no_grad():
            outputs_x, outputs_xd = model(source_frames, flows=flows, frame_names=frame_names)
        print(f"Output stats - min: {outputs_x.min():.3f}, max: {outputs_x.max():.3f}, mean: {outputs_x.mean():.3f}")
        print(f"Mask stats - min: {outputs_xd.min():.3f}, max: {outputs_xd.max():.3f}")
        print(f"Contains NaN: {torch.isnan(outputs_x).any()}, Contains Inf: {torch.isinf(outputs_x).any()}")

        # 如果包含NaN，打印更多信息
        if torch.isnan(outputs_x).any():
            print("ERROR: Model output contains NaN!")
            # 检查输入
            print(f"Input stats - min: {source_frames.min():.3f}, max: {source_frames.max():.3f}")
        last_smoke_mask = smoke_masks[:, -1].clone()

        return outputs_x, outputs_xd, last_smoke_mask

    with torch.no_grad():
        if mode == "virtual":
            for idx, batch in enumerate(test_loader):
                try:
                    sequence_id = idx
                    forward_start_time = time.time()
                    source_frames = torch.stack([frame.cuda() for frame in batch["sequence_source"]], dim=1)
                    target_frames = torch.stack([frame.cuda() for frame in batch["sequence_target"]], dim=1)
                    sequence_names = batch["sequence_names"]
                    flows = None
                    if "flows" in batch:
                        flows = batch["flows"]

                    B, T, C, H, W = source_frames.shape

                    all_outputs_x = []
                    all_outputs_xd = []
                    for t in range(1, T):  # t = 1,2,3,4
                        # 构建输入帧
                        if t == 1:
                            current_input = source_frames[:, :2]  # [帧0, 帧1]
                        else:
                            indices = [0, t - 1, t]  # [帧0, 前一帧, 当前帧]
                            current_input = source_frames[:, indices]

                        current_flows = None
                        if flows is not None:
                            current_flows = {}

                            # 相邻帧光流：取第 t-1 个
                            if 'adjacent' in flows and isinstance(flows['adjacent'], list):
                                if len(flows['adjacent']) >= t:
                                    flow_adj = flows['adjacent'][t - 1]
                                    if flow_adj is not None:
                                        if not flow_adj.is_cuda:
                                            flow_adj = flow_adj.cuda()
                                        current_flows['adjacent'] = flow_adj

                            if t >= 2 and 'long_range' in flows and isinstance(flows['long_range'], list):
                                long_idx = t - 2
                                if len(flows['long_range']) > long_idx:
                                    flow_long = flows['long_range'][long_idx]
                                    if flow_long is not None:
                                        if not flow_long.is_cuda:
                                            flow_long = flow_long.cuda()
                                        current_flows['long_range'] = flow_long

                            # 反向相邻帧光流
                            if 'adjacent_backward' in flows and isinstance(flows['adjacent_backward'], list):
                                if len(flows['adjacent_backward']) >= t:
                                    flow_adj_back = flows['adjacent_backward'][t - 1]
                                    if flow_adj_back is not None:
                                        if not flow_adj_back.is_cuda:
                                            flow_adj_back = flow_adj_back.cuda()
                                        current_flows['adjacent_backward'] = flow_adj_back

                            # 反向长程光流
                            if t >= 2 and 'long_range_backward' in flows and isinstance(flows['long_range_backward'], list):
                                long_idx = t - 2
                                if len(flows['long_range_backward']) > long_idx:
                                    flow_long_back = flows['long_range_backward'][long_idx]
                                    if flow_long_back is not None:
                                        if not flow_long_back.is_cuda:
                                            flow_long_back = flow_long_back.cuda()
                                        current_flows['long_range_backward'] = flow_long_back

                        with torch.no_grad():
                            if args.adaptive_weights:
                                output_x, output_xd, _ = inference_with_adaptive_weights(
                                    model, current_input, current_flows, sequence_names
                                )
                            else:
                                output_x, output_xd = model(
                                    current_input, flows=current_flows,
                                    frame_names=sequence_names, epoch_num=None
                                )

                        output_x = output_x.squeeze(0).squeeze(0)  # [C, H, W]
                        output_xd = output_xd.squeeze(0).squeeze(0)  # [1, H, W]

                        all_outputs_x.append(output_x)
                        all_outputs_xd.append(output_xd)

                    batch_frames = len(all_outputs_x)
                    frame_count += batch_frames
                    perf_stats['forward_pass'] += time.time() - forward_start_time
                    postproc_start = time.time()

                    for t_idx, (output, pred_mask) in enumerate(zip(all_outputs_x, all_outputs_xd)):
                        frame_id = t_idx + 2  # 实际帧号：2,3,4,5（跳过第1帧）
                        target = target_frames[0, t_idx + 1]  # target索引：1,2,3,4

                        # 处理输出图像
                        output_img = (output.clamp(-1, 1) * 0.5 + 0.5).cpu().numpy()
                        if output_img.shape[0] > 3:
                            output_img = output_img[:3]
                        output_img = np.transpose(output_img, (1, 2, 0))

                        # 处理GT图像
                        target_img = (target.clamp(-1, 1) * 0.5 + 0.5).cpu().numpy()
                        if target_img.shape[0] > 3:
                            target_img = target_img[:3]
                        target_img = np.transpose(target_img, (1, 2, 0))

                        # 处理掩码
                        pred_mask_img = pred_mask.cpu().numpy()
                        if pred_mask_img.ndim == 3 and pred_mask_img.shape[0] == 1:
                            pred_mask_img = pred_mask_img[0]

                        # 计算指标
                        output_img_rgb = output_img[..., ::-1]
                        target_img_rgb = target_img[..., ::-1]
                        psnr_value, ssim_value, mse_value = calculate_metrics(output_img_rgb, target_img_rgb)

                        metrics_data['sequence_id'].append(sequence_id)
                        metrics_data['frame_id'].append(frame_id)
                        metrics_data['psnr'].append(psnr_value)
                        metrics_data['ssim'].append(ssim_value)
                        metrics_data['mse'].append(mse_value)

                        safe_print(
                            f"Sequence {sequence_id}, Frame {frame_id}: PSNR={psnr_value:.2f}, SSIM={ssim_value:.4f}")

                        # 保存结果
                        save_start = time.time()
                        # ========== 修改点：取正确的帧名（t_idx+1 对应处理的帧） ==========
                        original_name = sequence_names[t_idx + 1] if isinstance(sequence_names[t_idx + 1], str) else \
                            sequence_names[t_idx + 1][0]
                        if not original_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            original_name += '.png'

                        img_array = (output_img_rgb * 255).astype(np.uint8)
                        Image.fromarray(img_array).save(os.path.join(desmoking_dir, original_name))

                        mask_arr = (pred_mask_img * 255).astype(np.uint8)
                        Image.fromarray(mask_arr).save(os.path.join(mask_dir, original_name))

                        perf_stats['saving'] += time.time() - save_start

                    perf_stats['postprocessing'] += time.time() - postproc_start

                    if (idx + 1) % 50 == 0:
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error processing sequence {idx}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

        else:  # mode == "real"
            last_output_frame = None
            global_frame_index = 0

            for idx, batch in enumerate(test_loader):
                try:
                    sequence_id = idx
                    print(f"\nProcessing sequence {sequence_id}")

                    # 获取原始frames
                    source_frames = [frame.cuda() for frame in batch["sequence_source"]]
                    target_frames = [frame.cuda() for frame in batch["sequence_target"]]
                    sequence_names = batch["sequence_names"]

                    # 处理维度
                    source_frames = [f.squeeze(0) if f.dim() == 4 and f.shape[0] == 1 else f for f in source_frames]
                    target_frames = [f.squeeze(0) if f.dim() == 4 and f.shape[0] == 1 else f for f in target_frames]

                    # 如果有上一序列的输出，替换第一帧
                    if sequence_id > 0 and last_output_frame is not None:
                        print(f"Using last output frame as first frame")
                        source_frames[0] = last_output_frame.clone()

                    B = 1  # batch size
                    T = len(source_frames)

                    # 存储当前序列的所有输出
                    sequence_outputs_x = []
                    sequence_outputs_xd = []

                    # 逐帧处理（从第2帧开始）
                    for t in range(1, T):
                        # 准备输入帧
                        if t == 1:
                            # 第2帧：使用帧1和帧2
                            current_frames = [source_frames[0], source_frames[1]]
                        else:
                            # 后续帧：使用第1帧、前一帧、当前帧
                            current_frames = [source_frames[0], source_frames[t - 1], source_frames[t]]

                        # 创建输入tensor
                        current_input = torch.stack(current_frames, dim=0).unsqueeze(0)  # [1, 2or3, C, H, W]

                        # 计算必要的光流
                        # 计算必要的光流
                        current_flows = None
                        if not args.disable_flow and gmflow_model is not None:
                            flow_start_time = time.time()
                            current_flows = {
                                'adjacent': [],
                                'long_range': [],
                                'adjacent_backward': [],
                                'long_range_backward': []
                            }

                            # 1. Adjacent光流：(t-1) → t（前一帧到当前帧）
                            frame_prev = source_frames[t - 1].unsqueeze(0)  # 前一帧
                            frame_curr = source_frames[t].unsqueeze(0)  # 当前帧

                            # 正向光流：prev → curr
                            flow_adj = compute_optical_flow(gmflow_model, frame_prev, frame_curr)
                            current_flows['adjacent'] = flow_adj

                            # 反向光流：curr → prev
                            flow_adj_back = compute_optical_flow(gmflow_model, frame_curr, frame_prev)
                            current_flows['adjacent_backward'] = flow_adj_back

                            if t >= 2:
                                frame_ref = source_frames[0].unsqueeze(0)  # 参考帧（第一帧）

                                # 正向光流：ref → curr
                                flow_long = compute_optical_flow(gmflow_model, frame_ref, frame_curr)
                                current_flows['long_range'] = flow_long

                                # 反向光流：curr → ref
                                flow_long_back = compute_optical_flow(gmflow_model, frame_curr, frame_ref)
                                current_flows['long_range_backward'] = flow_long_back

                            perf_stats['flow_computation'] += time.time() - flow_start_time

                        # 前向传播
                        forward_start_time = time.time()
                        with torch.no_grad():
                            output_x, output_xd = model(current_input, flows=current_flows,
                                                        frame_names=sequence_names, epoch_num=None)
                        perf_stats['forward_pass'] += time.time() - forward_start_time

                        # 提取输出（移除批次和时间维度）
                        output_x = output_x.squeeze(0).squeeze(0)  # [C, H, W]
                        output_xd = output_xd.squeeze(0).squeeze(0)  # [1, H, W]

                        sequence_outputs_x.append(output_x)
                        sequence_outputs_xd.append(output_xd)

                        # 保存最后一帧用于下一个序列
                        if t == T - 1:
                            last_output_frame = output_x.clone()

                    # 处理和保存所有输出帧
                    postproc_start = time.time()
                    for t_idx, (output, mask) in enumerate(zip(sequence_outputs_x, sequence_outputs_xd)):
                        frame_idx = t_idx + 1  # 实际帧索引（跳过第1帧）
                        target = target_frames[frame_idx]

                        # 处理输出
                        output_img = (output.clamp(-1, 1) * 0.5 + 0.5).cpu().numpy()
                        if output_img.shape[0] > 3:
                            output_img = output_img[:3]
                        output_img = np.transpose(output_img, (1, 2, 0))

                        # 处理目标
                        target_img = (target.clamp(-1, 1) * 0.5 + 0.5).cpu().numpy()
                        if target_img.shape[0] > 3:
                            target_img = target_img[:3]
                        target_img = np.transpose(target_img, (1, 2, 0))

                        # 处理掩码
                        pred_mask_img = mask.cpu().numpy()
                        if pred_mask_img.ndim == 3 and pred_mask_img.shape[0] == 1:
                            pred_mask_img = pred_mask_img[0]

                        # 计算指标
                        output_img_rgb = output_img[..., ::-1]
                        target_img_rgb = target_img[..., ::-1]
                        psnr_value, ssim_value, mse_value = calculate_metrics(output_img_rgb, target_img_rgb)

                        # 记录指标
                        actual_frame_id = global_frame_index + frame_idx
                        metrics_data['sequence_id'].append(sequence_id)
                        metrics_data['frame_id'].append(actual_frame_id)
                        metrics_data['psnr'].append(psnr_value)
                        metrics_data['ssim'].append(ssim_value)
                        metrics_data['mse'].append(mse_value)

                        print(
                            f"Sequence {sequence_id}, Frame {actual_frame_id}: PSNR={psnr_value:.2f}, SSIM={ssim_value:.4f}")

                        # 保存结果
                        save_start = time.time()
                        original_name = sequence_names[frame_idx] if isinstance(sequence_names[frame_idx], str) else \
                        sequence_names[frame_idx][0]
                        if not original_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            original_name += '.png'

                        img_array = (output_img_rgb * 255).astype(np.uint8)
                        Image.fromarray(img_array).save(os.path.join(desmoking_dir, original_name))

                        mask_arr = (pred_mask_img * 255).astype(np.uint8)
                        Image.fromarray(mask_arr).save(os.path.join(mask_dir, original_name))

                        perf_stats['saving'] += time.time() - save_start
                        frame_count += 1

                    perf_stats['postprocessing'] += time.time() - postproc_start

                    # 更新全局帧索引
                    global_frame_index += (T - 1)  # 跳过了第一帧

                    if (idx + 1) % 10 == 0:
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error processing sequence {idx}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    global_frame_index += 4  # stride
                    continue

    perf_stats['total'] = time.time() - total_start
    perf_stats['frame_count'] = frame_count
    print("\n Statistics on assessment indicators ")

    metrics_df = pd.DataFrame(metrics_data)
    avg_psnr = metrics_df['psnr'].mean()
    avg_ssim = metrics_df['ssim'].mean()
    avg_mse = metrics_df['mse'].mean()

    print(f"average index:")
    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average MSE: {avg_mse:.6f}")

    if mode == "virtual":
        frame_stats = metrics_df.groupby('frame_id').agg({
            'psnr': ['mean', 'std', 'min', 'max'],
            'ssim': ['mean', 'std', 'min', 'max']
        })
        print("\n Indicators for evaluating the position of each frame.")
        print(frame_stats)
    else:
        print("\n real scenes processed continuously without individual frame position statistics")

    metrics_df.to_csv(os.path.join(metrics_dir, 'frame_metrics.csv'), index=False)

    print("\n======== Performance Analysis Report ========")
    print(f"Scene mode: {mode}")
    print(f"Total Frames: {perf_stats['frame_count']}")
    print(f"Total inference time: {perf_stats['total']:.2f}seconds")
    if mode == "real":
        print(f"Real-time timeline calculation time: {perf_stats['flow_computation']:.2f}seconds")
    print(f"Average FPS: {perf_stats['frame_count'] / perf_stats['total']:.2f}")

    print("\n reasoning complete! The results have been saved to {}".format(output_dir))
    return metrics_df, perf_stats


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Video Smoke Removal Inference - Supporting Real/Virtual Scenes')
    parser.add_argument('--mode', type=str, choices=['real', 'virtual'], default='real',
                        help='Inference mode - real: Real scene (real-time optical flow calculation), virtual: Virtual scene (pre-calculated optical flow)')
    parser.add_argument('--root_path', type=str, default='E:/1/PSSv1',
                        help='Test data root directory')
    parser.add_argument('--output_dir', type=str, default='D:/dsu-0802/validation_results',
                        help='Results save directory')
    parser.add_argument('--model_path', type=str,
                        default="D:/dsu-0802/Checkpoints/best_psnr_epoch_134_good_2.pth",
                        help='Model path')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--cfg', type=str, default="./configs/deswin_patch4_windows8_256.yaml",
                        help='Configuration file path')
    parser.add_argument('--sequence_length', type=int, default=5,
                        help='Multi-frame sequence length, consistent with training')
    parser.add_argument('--flow_dir', type=str, default="E:/12.9/PSSv1-11/flow_npy",
                        help='Pre-calculated optical flow directory for virtual scene')
    parser.add_argument('--gmflow_checkpoint', type=str,
                        default="D:/StereoVideo/gmflow/checkpoints/gmflow_sintel-0c07dcb3.pth",
                        help='GMFlow pre-trained model path')
    parser.add_argument('--opts', default=None, nargs='+', help='Other configuration options')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--zip', action='store_true', help='Use compressed dataset')
    parser.add_argument('--cache-mode', type=str, default='part', help='Cache mode')
    parser.add_argument('--resume', help='Resume checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help='Gradient accumulation steps')
    parser.add_argument('--use-checkpoint', action='store_true', help='Whether to use gradient checkpoint')
    parser.add_argument('--amp-opt-level', type=str, default='O1', help='Mixed precision optimization level')
    parser.add_argument('--tag', help='Experiment tag')
    parser.add_argument('--eval', action='store_true', help='Only perform evaluation')
    parser.add_argument('--throughput', action='store_true', help='Test throughput')
    parser.add_argument('--disable_edge', action='store_true', help='禁用边缘感知增强模块')
    parser.add_argument('--adaptive_weights', action='store_true', help='使用基于烟雾密度的自适应权重')
    parser.add_argument('--disable_temporal', action='store_true', help='禁用参考帧引导的时序一致性学习')
    parser.add_argument('--disable_flow', action='store_true', help='禁用光流引导')
    parser.add_argument('--disable_attention', action='store_true', help='禁用注意力融合')
    args = parser.parse_args()

    # Load DeSwin configuration
    from config import get_config

    config = get_config(args)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n>> Video Smoke Removal Efficient Inference - {args.mode} Scene Mode")
    print(f">> Model file: {args.model_path}")
    print(f">> Output directory: {args.output_dir}")
    if args.mode == "real":
        print(f">> GMFlow model: {args.gmflow_checkpoint}")
    else:
        print(f">> Pre-calculated optical flow directory: {args.flow_dir}")
    print("\n")

    # Use improved inference function
    metrics_df, perf_stats = inference_with_adaptive_mode(
        args.model_path,
        config,
        args.root_path,
        args.output_dir,
        mode=args.mode,
        sequence_length=args.sequence_length,
        flow_dir=args.flow_dir,
        gmflow_checkpoint=args.gmflow_checkpoint
    )