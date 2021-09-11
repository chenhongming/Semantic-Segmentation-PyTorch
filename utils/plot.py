import os
import math
import matplotlib.pyplot as plt


class Writer:
    """Save training avg loss to log file and draw a curve."""

    def __init__(self, log_path):
        self.file = None
        self.log_path = log_path
        if self.log_path is not None:
            self.log_file = os.path.join(self.log_path, 'log.txt')
            self.file = open(self.log_file, 'w')
            self.file.write('Epoch' + '\t' + 'Train Loss' + '\n')

    def append(self, msg):
        assert len(msg) == 2
        self.file.write(str(msg[0]) + '\t' + '\t' + "{0:.6f}".format(msg[1]) + '\n')
        self.file.flush()

    def draw_curve(self, arch):
        curve_path = '{}/{}_curve.pdf'.format(self.log_path, arch)
        train_loss = []
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # exclude first line
                if line.startswith('E'):
                    continue
                train_loss.append(float(line.strip().split('\t\t')[1]))

        plt.ylim(0, math.ceil(float(train_loss[0])))
        x = [i for i in range(1, len(lines))]
        plt.xticks(x)
        plt.xlabel('epochs')
        plt.ylabel('train ave loss')

        plt.plot(x, train_loss, color='red')
        plt.legend(['{}-train-loss'.format(arch)])
        plt.grid(True)

        plt.savefig(curve_path, dpi=300, format='pdf')
        plt.close()
