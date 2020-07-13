from torchvision import transforms
def transform():
    return transforms.Compose([
                        transforms.Grayscale(1),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomErasing(0.2,(0.2,0.2)),
                        transforms.ToTensor(),
                    ])