import torch.optim as optim

def optimizer(model,config):
    return optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )