import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if loss doesn't improve after a given patience."""
    def __init__(self, patience: int = 10, verbose: bool = False, saving_checkpoint: bool = True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            saving_checkpoint (bool): If True, will save checkpoint of the best model
        """
        self.patience = patience
        self.verbose = verbose
        self.saving_checkpoint = saving_checkpoint
        self._counter = 0
        self._best_score = None
        self._last_loss = None
        self.delta = 0.1
        self.early_stop = False
        self._loss_min = np.Inf
        self.checkpoint_path = "checkpoint.pt"
    
    def __str__(self):
        this = "EarlyStopping: (\n"
        this += f"\t Patience: {self.patience} \n"
        this += f"\t SavingCheckpoint: {self.saving_checkpoint} \n"
        this += f"\t CheckpointPath: {self.checkpoint_path} \n"
        this += ')'
        return this

    def __call__(self, loss, model):

        if self._last_loss is None:
            current_delta = np.Inf
        else:
            current_delta = np.abs(((self._last_loss - loss) * 100) / self._last_loss)
        # print(current_delta)

        if current_delta < self.delta:
            self._counter += 1
            print(f'EarlyStopping counter: {self._counter} out of {self.patience}') if self.verbose else 0
            if self._counter >= self.patience:
                self.early_stop = True
        else:
            self._counter = 0

        score = -loss

        if self._best_score is None:
            self._best_score = score
            self.save_checkpoint(loss, model) if self.saving_checkpoint else 0
        elif score < self._best_score:
            self._best_score = score
            self.save_checkpoint(loss, model) if self.saving_checkpoint else 0

        self._last_loss = loss

    def save_checkpoint(self, loss, model):
        '''
            Saves model when loss decrease.
        '''
        if self.verbose:
            print(f'Loss decreased ({self._loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self._loss_min = loss

    def reset(self):
        self._counter = 0
        self._best_score = None
        self.early_stop = False
        self._loss_min = np.Inf


if __name__ == '__main__':
    es = EarlyStopping(patience=2, verbose=True, saving_checkpoint=False)
    losses = [100, 70, 85, 68, 67, 67.01, 67.02, 100, 101, 68, 68.001, 68.01, 68.1, 68.01, 68.001, 68.0001, 68.00001]
    for loss in losses:
        print(loss)
        es(loss, None)
        print()
    print(es.early_stop)
