from torchvision import transforms
def transform():
    return transforms.Compose([
                        transforms.Grayscale(1),
                        HistogramEqualization(),
                        transforms.RandomRotation(30),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomCrop(40),
                        transforms.Resize(48),
                        transforms.ToTensor(),
                    ])