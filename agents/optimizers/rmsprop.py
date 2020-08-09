import torch.optim as optim

def optimizer(model,config):
    return optim.RMSprop(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay

    )