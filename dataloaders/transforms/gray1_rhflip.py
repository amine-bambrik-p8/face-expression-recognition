from torchvision import transforms
def transform():
    return transforms.Compose([
                        transforms.Grayscale(1),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor(),
                    ])