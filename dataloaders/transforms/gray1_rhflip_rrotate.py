from torchvision import transforms
def transform():
    return transforms.Compose([
                        transforms.Grayscale(1),
                        transforms.RandomRotation(30),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.CenterCrop(42),
                        transforms.Resize(48),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5],[0.5])
                    ])