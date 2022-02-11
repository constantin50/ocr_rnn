import torch

ALPHABET = "%(),-./0123456789:;?[]«»АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё-"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
REPORT_ACCURACY = False
PATH_TO_IMGDIR = "/content/images/"
PATH_TO_LABELS = "/content/labels/"
