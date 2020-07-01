import torch.optim as optim

def optimizer(model,config):
    return optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
    )