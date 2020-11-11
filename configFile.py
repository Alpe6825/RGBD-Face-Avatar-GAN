# Data
DatasetName = "Philipp-Known-Setting-RGBDFaceAvatarGAN-Datensatz03-09-2020" #"MelSpecDataset" ##"Eva-RGBDFaceAvatarGAN-Datensatz23-08-2020" #3.Durchlauf-4.Datensatz-OhneLampeImHintergrundUndOhneStuhllehne" #"Carina-RGBDFaceAvatarGAN-Datensatz01-09-2020" #
FlipYAxis = False
IMAGE_SIZE = 512

# Pix2Pix
INPUT_CHANNEL = 1

# Depth Image
DEPTH_OFFSET = 300
DEPTH_MAX = 256 - 1

# IREyeTracking
IRET_Region = {
    "x": 500,
    "y": 320,
    "width": 160,
    "height": 40
}
IRET_THRESHOLD = 88

# Test
TEST_INPUT = "Camera" # or "OSC"
USE_FLC = True
