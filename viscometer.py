# requires model that provides inference data and saves it in the dataset/realfluid/answer

# Inference for WebGL Segmentation
import cv2
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from preprocess.mobile_sam import sam_model_registry, SamPredictor

transform = A.Compose(
    [A.Normalize(max_pixel_value=255.0), ToTensorV2()], is_check_shapes = False) # disable size match verificaiton between img and mask

vortex_model = sam_model_registry["vit_t"](checkpoint="preprocess/model_seg/Vortex0219_01.pth")
vortex_model.eval()
predictor = SamPredictor(vortex_model)

test_image = cv2.imread("test_seg.jpg", cv2.COLOR_BGR2RGB)
test_image = test_image[0:1024, 0:1024]

vortex_model.cuda()
transformed = transform(image=test_image)['image']
batched_input = [{"image": transformed.cuda(), "original_size": (256, 256)}] 
pred_mask = vortex_model(batched_input=batched_input, multimask_output=False)[0]["masks"]
pred_mask = pred_mask.squeeze().detach().cpu().numpy()

plt.imsave("test_mask.jpg", pred_mask, cmap="gray")