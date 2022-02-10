import torch

ALPHABET = "абвгдежзийклмнопрстуфхцчшщъыьэюяё-"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
REPORT_ACCURACY = False
PATH_TO_IMGDIR = "/content/images/"
PATH_TO_LABELS = "/content/labels/"