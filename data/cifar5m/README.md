1. Download part{0,...,5}.npz into this directory via instructions at
   https://github.com/preetum/cifar5m.
   You can use something like [cliget](https://addons.mozilla.org/en-US/firefox/addon/cliget/)
   to get links to download the files from your browser.
2. Run the following commands from the root of this repository:
   ```
   python -m src.ax.data.convert_cifar5m
   python -m src.ax.data.convert_cifar10
   ```
   The largest split `train.beton` has 5,942,688 samples.
