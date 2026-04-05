import os
import math
import itertools
from tqdm import tqdm
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from diffusers import DiffusionPipeline, DDIMScheduler

from utils import *

# main 함수
def main(args):
    set_random_seed(42)
    project_root = Path(__file__).resolve().parent.parent
    save_dir = project_root / f"{args.output_dir}/{args.dataset_id}/{args.wm_type}"
    inverted_path = os.path.join(save_dir, "inverted_latents")
    os.makedirs(inverted_path, exist_ok=True)

    # [Datasets]
    meta_annot, prompt_key, gt_folder = get_text_dataset(args.dataset_id)

    # [Attack Settings]
    case_names = ["Clean", "Brightness", "Contrast", "JPEG", "Blur", "Noise", "BM3D",
              "VAE-B", "VAE-C", "Diff", "CC", "RC", "Avg"]
    attack_dict = {"Brightness":6, "Contrast":0.5, "JPEG":25, "Blur":5, "Noise":0.05, "BM3D":0.1, 
                "VAE-B":3, "VAE-C":3,
                "CC":0.5, "RC":0.7}

    # [Evaluation Settings]
    num_dataset = len(meta_annot)
    identify_gt_indices = np.load(os.path.join(save_dir, f"identify_gt_indices_{num_dataset}.npy"))
    detect_trials = 1000
    RANGE_EVAL = range(0,detect_trials)
    
    # [Stable-Diffusion-v2-1-base Settings]
    model_id = "stabilityai/stable-diffusion-2-1-base"
    resolution = 512
    torch_dtype = torch.float32

    # [Load Stable-Diffusion pipeline]
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # [Result list]
    no_watermark_results_list = []
    Fourier_watermark_results_list = []
    id_acc_results_list = []
    save_verify_name = "verify-l1.npz"
    save_identify_name = "identify-acc.npz"

    with torch.no_grad():
        # [Load Ground-Truth patterns / OSS 参数]
        if args.wm_type == "OSS":
            from oss import oss_extract
            # OSS 使用比特流匹配，不需要加载 pattern_list
            oss_num_bits = math.ceil(math.log2(wm_capacity))  # 11 比特编码 2048 个身份
            oss_seed = w_seed  # 与 generate.py 中相同的码片生成种子
            oss_msg_list = np.load(os.path.join(save_dir, f"oss_msg_list_{num_dataset}.npy"))
            save_verify_name = "verify-ber.npz"
        else:
            Fourier_watermark_pattern_list = torch.load(os.path.join(save_dir, f"pattern_list-{wm_capacity}.pt")).cpu()

            # [Evaluation methods]
            is_center = args.wm_type in ["HSTR", "HSQR"]
            channel_min = args.wm_type in ["RingID", "HSTR"]
            if args.wm_type in ["Tree-Ring", "RingID", "HSTR"]:
                eval_method = {"Distance":"L1", "Metrics":"|a-b|", "func":get_distance, 
                    "kwargs":{"p":1, "center": is_center, 
                        "mode":"complex", "channel_min":channel_min}}
            elif args.wm_type == "HSQR":
                eval_method = {"Distance":"L1", "Metrics":"|a-b|", "func":get_distance_hsqr, 
                    "kwargs":{"p":1, "center": is_center}}
            
            # [Detection Area] Channel and Mask
            channel = RINGID_WATERMARK_CHANNEL if args.wm_type in ["RingID", "HSTR"] else TREE_WATERMARK_CHANNEL
            if args.wm_type == "Tree-Ring":
                mask = watermark_region_mask_tree.cpu() # (1,64,64), r=14 inner-circle mask (boolean)
            elif args.wm_type == "RingID":
                mask = watermark_region_mask_ringid.cpu() # (C_R+C_H,64,64)
            elif args.wm_type == "HSTR":
                mask = watermark_region_mask_hstr.cpu() # (C_R+C_H,64,64)
        
        print("Attack-Detection Starts")
        for idx in tqdm(RANGE_EVAL):
            key_index = identify_gt_indices[idx]
            if args.wm_type != "OSS":
                pattern_gt = Fourier_watermark_pattern_list[key_index].cpu()
            # Set random seed
            this_seed = 42 + idx
            set_random_seed(this_seed)
            # File inputs
            file_name = f"{idx}.png"

            # [Attack]
            img_pil = Image.open(os.path.join(save_dir, f"img_pil/{file_name}"))
            img_pil_wm = Image.open(os.path.join(save_dir, f"img_pil_wm/{file_name}"))
            img_pil_diff_attacked = Image.open(os.path.join(save_dir, f"img_pil-diffatt_fp16/{file_name}"))
            img_pil_wm_diff_attacked = Image.open(os.path.join(save_dir, f"img_pil_wm-diffatt_fp16/{file_name}"))
            distorted_image_list = [
                [img_pil, img_pil_wm], # Clean
                image_distortion(img_pil, img_pil_wm, seed=this_seed, brightness_factor=attack_dict["Brightness"]),
                image_distortion(img_pil, img_pil_wm, seed=this_seed, contrast_factor=attack_dict["Contrast"]),
                image_distortion(img_pil, img_pil_wm, seed=this_seed, jpeg_ratio=attack_dict["JPEG"]),
                image_distortion(img_pil, img_pil_wm, seed=this_seed, gaussian_blur_r=attack_dict["Blur"]),
                image_distortion(img_pil, img_pil_wm, seed=this_seed, gaussian_std=attack_dict["Noise"]),
                image_distortion(img_pil, img_pil_wm, seed=this_seed, bm3d_sigma=attack_dict["BM3D"]),
                image_distortion(img_pil, img_pil_wm, seed=this_seed, vaeb_quality=attack_dict["VAE-B"]),
                image_distortion(img_pil, img_pil_wm, seed=this_seed, vaec_quality=attack_dict["VAE-C"]),
                [img_pil_diff_attacked, img_pil_wm_diff_attacked], # Diffusion-Attack (Regeneration)
                image_distortion(img_pil, img_pil_wm, seed=this_seed, center_crop_area_ratio=attack_dict["CC"]),
                image_distortion(img_pil, img_pil_wm, seed=this_seed, random_crop_area_ratio=attack_dict["RC"]),]
            
            img_pil_distorted_list = [pair[0] for pair in distorted_image_list]
            img_pil_wm_distorted_list = [pair[1] for pair in distorted_image_list]

            # [Save DDIM-Inverted Latents]
            no_wm_distorted_zT = ddim_invert(pipe, img_pil_distorted_list, invert_guidance=0).cpu() # (N_attack,4,64,64)
            Fourier_wm_distorted_zT = ddim_invert(pipe, img_pil_wm_distorted_list, invert_guidance=0).cpu() # (N_attack,4,64,64)
            np.save(os.path.join(inverted_path, f"{idx}-no_latents.npy"), no_wm_distorted_zT.numpy())
            np.save(os.path.join(inverted_path, f"{idx}-wm_latents.npy"), Fourier_wm_distorted_zT.numpy())

            if args.wm_type == "OSS":
                # ---- OSS 检测分支 ----
                gt_msg = oss_msg_list[idx].tolist()  # 该图像的 ground-truth 比特消息

                # [Verification] 使用 BER 作为验证指标
                # 约定：BER 越低表示越可能含水印 → 取负 BER 作为"得分"，与其他方法的 -L1 语义一致
                no_wm_result = []
                Fourier_wm_result = []
                for distortion_index in range(len(distorted_image_list)):
                    # 无水印图像：提取比特并计算 BER
                    no_wm_zT = no_wm_distorted_zT[distortion_index][None, ...]  # (1,4,64,64)
                    no_wm_bits = oss_extract(no_wm_zT, oss_seed, oss_num_bits)
                    no_wm_ber = sum(a != b for a, b in zip(no_wm_bits, gt_msg)) / oss_num_bits
                    no_wm_result.append(-no_wm_ber)  # 取负值，与 -L1 语义一致

                    # 有水印图像：提取比特并计算 BER
                    wm_zT = Fourier_wm_distorted_zT[distortion_index][None, ...]  # (1,4,64,64)
                    wm_bits = oss_extract(wm_zT, oss_seed, oss_num_bits)
                    wm_ber = sum(a != b for a, b in zip(wm_bits, gt_msg)) / oss_num_bits
                    Fourier_wm_result.append(-wm_ber)  # 取负值

                no_watermark_results_list.append(no_wm_result)
                Fourier_watermark_results_list.append(Fourier_wm_result)

                # [Identification] 比特流内容匹配：提取比特 → 转换为索引 → 与 ground-truth 索引比较
                id_acc_result = []
                for distortion_index in range(len(distorted_image_list)):
                    wm_zT = Fourier_wm_distorted_zT[distortion_index][None, ...]  # (1,4,64,64)
                    wm_bits = oss_extract(wm_zT, oss_seed, oss_num_bits)
                    # 比特串 → 整数索引
                    extracted_index = 0
                    for b in wm_bits:
                        extracted_index = (extracted_index << 1) | b
                    id_acc = (extracted_index == key_index)
                    id_acc_result.append(id_acc)
                id_acc_results_list.append(id_acc_result)

            else:
                # ---- 非 OSS 方法：原有 FFT + L1 距离检测逻辑 ----
                # Latent Fourier
                if args.wm_type in ["Tree-Ring", "RingID"]:
                    no_wm_distorted_zT_fft = fft(no_wm_distorted_zT)
                    Fourier_wm_distorted_zT_fft = fft(Fourier_wm_distorted_zT)
                elif args.wm_type in ["HSTR", "HSQR"]:
                    no_wm_distorted_zT_fft = torch.zeros_like(no_wm_distorted_zT, dtype=torch.complex64)
                    Fourier_wm_distorted_zT_fft = torch.zeros_like(Fourier_wm_distorted_zT, dtype=torch.complex64)
                    no_wm_distorted_zT_fft[center_slice] = fft(no_wm_distorted_zT[center_slice])
                    Fourier_wm_distorted_zT_fft[center_slice] = fft(Fourier_wm_distorted_zT[center_slice])

                # [Verification] Calculating L1 distances - for ROC Curves (TPR, FPR, Thresholds)
                no_wm_result = []
                Fourier_wm_result = []
                for distortion_index in range(len(distorted_image_list)):
                    no_wm_zT_fft = no_wm_distorted_zT_fft[distortion_index][None, ...]
                    Fourier_wm_zT_fft = Fourier_wm_distorted_zT_fft[distortion_index][None, ...]
                    if args.wm_type in ["Tree-Ring", "RingID", "HSTR"]:
                        no_wm_verify_l1 = -eval_method['func'](pattern_gt, no_wm_zT_fft, mask=mask, channel=channel, **eval_method['kwargs'])
                        Fourier_wm_verify_l1 = -eval_method['func'](pattern_gt, Fourier_wm_zT_fft, mask=mask, channel=channel, **eval_method['kwargs'])
                    elif args.wm_type == "HSQR":
                        no_wm_verify_l1 = -eval_method['func'](pattern_gt, no_wm_zT_fft, channel=channel, **eval_method['kwargs'])
                        Fourier_wm_verify_l1 = -eval_method['func'](pattern_gt, Fourier_wm_zT_fft, channel=channel, **eval_method['kwargs'])
                    no_wm_result.append(no_wm_verify_l1)
                    Fourier_wm_result.append(Fourier_wm_verify_l1)
                no_watermark_results_list.append(no_wm_result)
                Fourier_watermark_results_list.append(Fourier_wm_result)

                # [Identification] Ground-Truth Pattern Matching Accuracy
                id_acc_result = []
                for distortion_index in range(len(distorted_image_list)):
                    Fourier_wm_zT_fft = Fourier_wm_distorted_zT_fft[distortion_index][None, ...]
                    candidate_distances_list = []
                    for Fourier_watermark_pattern in Fourier_watermark_pattern_list:
                        if args.wm_type in ["Tree-Ring", "RingID", "HSTR"]:
                            candidate_distance = eval_method['func'](Fourier_watermark_pattern, Fourier_wm_zT_fft, mask=mask, channel=channel, **eval_method['kwargs'])
                        elif args.wm_type == "HSQR":
                            candidate_distance = eval_method['func'](Fourier_watermark_pattern, Fourier_wm_zT_fft, channel=channel, **eval_method['kwargs'])
                        candidate_distances_list.append(candidate_distance)
                    id_acc = np.argmin(np.array(candidate_distances_list)) == key_index
                    id_acc_result.append(id_acc)
                id_acc_results_list.append(id_acc_result)

    # [Save results]
    no_watermark_results_list_array = np.array(no_watermark_results_list) # (1000,12)
    Fourier_watermark_results_list_array = np.array(Fourier_watermark_results_list) # (1000,12)
    id_acc_results_list_array = np.array(id_acc_results_list) # (1000,12)
    np.savez(os.path.join(save_dir, save_verify_name), 
        no_wm=no_watermark_results_list_array, 
        wm=Fourier_watermark_results_list_array)
    np.savez(os.path.join(save_dir, save_identify_name), 
        wm=id_acc_results_list_array)

    # [Print results]
    from prettytable import PrettyTable
    no_wms = no_watermark_results_list_array = np.array(no_watermark_results_list)
    wms = np.array(Fourier_watermark_results_list) # (1000,12)
    i_wms = np.array(id_acc_results_list) # (1000,12)
    
    auc_list, acc_list, low_list = [], [], []
    for idx in range(no_wms.shape[1]): # 12 Cases
        no_wm = no_wms[:, idx].tolist()
        wm = wms[:, idx].tolist()
        distances = no_wm + wm
        labels = [0] * len(no_wm) + [1] * len(wm)
        fpr, tpr, _ = metrics.roc_curve(labels, distances, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        acc = 1 - ((fpr + (1 - tpr)) / 2).min()
        low = tpr[np.where(fpr < 0.01)[0][-1]] if np.any(fpr < 0.01) else 0.0
        
        auc_list.append(auc)
        acc_list.append(acc)
        low_list.append(low)
    
    id_accs = np.mean(i_wms.squeeze(), axis=0).tolist()

    results = {}
    for metric_name, values in zip(
        ["AUC", "MaxAcc", "TPR@1%FPR", "Id-Acc"],
        [auc_list, acc_list, low_list, id_accs]
    ):
        results[metric_name] = values + [np.mean(values)]
    
    print()
    print("#" * 60)
    verify_label = "BER" if args.wm_type == "OSS" else "L1"
    print(f"Table 1: Verification Performance ({verify_label})")
    table = PrettyTable()
    table.field_names = ["WM Type"] + case_names
    table.add_row([args.wm_type] + [f"{v:.3f}" for v in results["TPR@1%FPR"]])
    print(table)
    print()

    print("#" * 60)
    print("Table 2: Identification Accuracy")
    table = PrettyTable()
    table.field_names = ["WM Type"] + case_names
    table.add_row([args.wm_type] + [f"{v:.3f}" for v in results["Id-Acc"]])
    print(table)
    print()
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--wm_type", choices=["Tree-Ring", "RingID", "HSTR", "HSQR", "OSS"], required=True, help="Choose watermarking methods")
    parser.add_argument("--dataset_id", choices=["coco", "Gustavo", "DB1k"], required=True, help="Choose dataset_id")
    parser.add_argument("--output_dir", default="outputs", help="output directory: ./[output_dir]/")
    args = parser.parse_args()
    main(args)
    