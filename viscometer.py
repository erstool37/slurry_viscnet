# Inference for Real Fluid Viscosity
### provides infernece data and saves it in the dataset/realfluid/answer



import cv2
from preprocess.mobile_sam import sam_model_registry, SamPredictor
from src.models.ViscosityEstimator import ViscosityEstimator

# 1. Inference for WebGL Segmentation
"""
vortex_model = sam_model_registry["vit_t"](checkpoint="preprocess/model_seg/Vortex0219_01.pth")
vortex_model.eval()
vortex_model.cuda()

#image setting
test_image = cv2.imread("test_seg.jpg")
test_image = test_image[0:1024, 0:1024] # required because predictor.py does not use postprocess yet

#predict
predictor = SamPredictor(vortex_model)
predictor.set_image(image=test_image, image_format="RGB") # normalize RGB, padding, make tensor
mask, _, _ = predictor.predict(multimask_output = False, return_logits=True) # get (256, 256) masked image with 0~255 uint8 format

cv2.imwrite("test_mask.jpg", mask)
"""

# 2. Inference for CFD Viscosity Estimation

with open("config_reg.yaml", "r") as file:
    config = yaml.safe_load(file)

CHECKPOINT = config["settings"]["checkpoint"] 
REAL_CHECKPOINT = config["settings"]["real_checkpoint"]
CNN = config["settings"]["cnn"]
LSTM_SIZE = int(config["settings"]["lstm_size"])
LSTM_LAYERS = int(config["settings"]["lstm_layers"])
FRAME_NUM = int(config["settings"]["frame_num"])
TIME = int(config["settings"]["time"])
OUTPUT_SIZE = int(config["settings"]["output_size"])

# test dataset
test_video = cv2.cap

visc_model = ViscosityEstimator(CNN, LSTM_SIZE, LSTM_LAYERS, OUTPUT_SIZE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
visc_model.load_state_dict(torch.load(CHECKPOINT))
visc_model.eval()
visc_model.cuda()

outputs = visc_model(frames)

with open(para_path, 'r') as file:
            data = json.load(file)
            density = data["density"]
            dynVisc = data["dynamic_viscosity"]
            surfT = data["surface_tension"]
# 
print("pred outputs", outputs)
print("ground_truth", density, dynVisc, surfT)
