from agents.generic import GenericAgent
import torch.optim as optim
class GenericAgentLRDecay(GenericAgent):
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self, config):
        super().__init__(config)

    def train(self):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,factor=self.config.lr_scheduler_factor,mode=self.config.lr_scheduler_mode,patience=2,min_lr=self.config.min_lr)
        for epoch in range(1, self.config.max_epoch + 1):
            (train_loss,train_accuracy) = self.train_one_epoch()
            if epoch % self.config.validate_every == self.config.validate_every-1:
                (loss,accuracy) = self.validate()
                scheduler.step(loss)
                self.summary_writer.add_scalars('accuracy', {
                        'training_{}'.format(self.config.optimizer):train_accuracy,
                        'validation_{}'.format(self.config.optimizer):valid_accuracy
                        }, epoch)
                self.summary_writer.add_scalars('loss', {
                        'training_{}'.format(self.config.optimizer):train_loss,
                        'validation_{}'.format(self.config.optimizer):loss
                        }, epoch)
                if self.config.do_save and (self.best_metric is None or self.best_metric > valid_loss):
                    self.logger.info('Saving Model with loss %f previous best loss was %f \n'% (loss, self.best_metric if self.best_metric is not None else 0.0))
                    self.best_metric = loss
                    self.save_checkpoint()
            self.current_epoch += 1