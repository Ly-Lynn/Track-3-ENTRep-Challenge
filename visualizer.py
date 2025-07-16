import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

class LossVisualizer:
    def __init__(self, history):
        self.train_loss = history.get('train_loss', [])
        self.val_loss = history.get('val_loss', [])
        
        # if not self.train_loss or not self.val_loss:
        #     raise ValueError("History must contain both 'train_loss' and 'val_loss' with non-empty lists.")
        
    def plot(self, title="Training and Validation Loss", save_path=None):
        epochs = range(1, len(self.train_loss) + 1)
        
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.train_loss, label='Train Loss', marker='o')
        plt.plot(epochs, self.val_loss, label='Validation Loss', marker='s')
        
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")