from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
writer=SummaryWriter
img_path="data/train/ants_image/0013035.jpg"
img = Image.open(img_path)
print(img)
trans_totensor = transforms.ToTensor()
img_tensor=trans_totensor(img)
writer.add_image("Totensor", img_tensor)
print(img_tensor[0][0][0])
trans_norm=transforms.Normalize([0.5, 0.5, 0.5],[0.5,0.5,0.5])
img_norm=trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.close()