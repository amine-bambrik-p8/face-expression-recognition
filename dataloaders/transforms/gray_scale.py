from torchvision import transforms
def transform():
    return transforms.Compose([
                    transforms.Grayscale(1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5),(0.5))
                ])