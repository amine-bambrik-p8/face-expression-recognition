import torch.optim as optim

def optimizer(model,config):
    return optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=[config.betas[0],config.betas[1]],
        weight_decay=config.weight_decay
    )