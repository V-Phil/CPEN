import os

from PIL import Image
from tqdm import tqdm

from cpen import cpen3
from utils.utils_metrics import compute_mIoU, show_results

import numpy as np  # ←可视化增强需要


if __name__ == "__main__":
    miou_mode = 0
    num_classes = 2
    name_classes = ["_background_", "crack"]
   
    VOCdevkit_path = 'VOCdevkit'

    image_ids       = open(os.path.join(VOCdevkit_path, "Test0717/ImageSets/Segmentation/val.txt"),'r').read().splitlines()
    gt_dir          = os.path.join(VOCdevkit_path, "Test0717/SegmentationClass/png/")
    miou_out_path   = "miou_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        cpen = cpen3(model_path="logs/best_epoch_weights.pth")
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "Test0717/JPEGImages/jpg/"+image_id+".jpg")
            image = Image.open(image_path)
            image = cpen.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes,
                                                        name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

    
    idx_crack = 1
    alpha = 0.5  # 红色遮罩透明度（0~1）

    
    image_dir = os.path.join(VOCdevkit_path, "Test0717/JPEGImages/jpg")
   
    vis_on_orig_dir = os.path.join(miou_out_path, 'colored-results-on-image')
    os.makedirs(vis_on_orig_dir, exist_ok=True)

    print("Creating overlay visualization on original images...")
    for image_id in tqdm(image_ids):
        mask_path = os.path.join(pred_dir, image_id + ".png")
        orig_img_path = os.path.join(image_dir, image_id + ".jpg")

        if not os.path.exists(mask_path) or not os.path.exists(orig_img_path):
            print(f"Warning: {image_id} missing mask or original image, skip")
            continue

        
        mask = np.array(Image.open(mask_path))
        orig_img = Image.open(orig_img_path).convert("RGB")
        orig_img = np.array(orig_img)

      
        if mask.shape[:2] != orig_img.shape[:2]:
            mask = np.array(Image.open(mask_path).resize((orig_img.shape[1], orig_img.shape[0]), Image.NEAREST))

       
        red_mask = np.zeros_like(orig_img)
        red_mask[mask == idx_crack] = [255, 0, 0]

    
        vis = orig_img.copy()
        vis[mask == idx_crack] = (
                (1 - alpha) * vis[mask == idx_crack] + alpha * red_mask[mask == idx_crack]
        )
        vis = vis.astype(np.uint8)
        vis_img = Image.fromarray(vis)
        vis_img.save(os.path.join(vis_on_orig_dir, image_id + ".jpg"))

    print(f"可视化叠加完毕，已保存至 {vis_on_orig_dir}")
