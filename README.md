## Setup

1. Create a conda environment:
   ```
   conda env create -f environment.yml
   ```
   If you are on macos, use `environment-macos.yml` instead.
2. Install pip dependencies:
   ```
   conda activate scaling-v2
   pip install -r requirements.txt
   ```

3. (Optional) If you run into an error of the form `GLIBCXX_3.4.26 not found`, you may want to augment your `LD_LIBRARY_PATH` with the conda environment `lib` folder. For example, Tony solved the error by adding the following to his `.bash_aliases` file:
   ```
   export LD_LIBRARY_PATH="/home/gridsan/twang/.conda/envs/scaling-v2/lib"
   ```

## Typechecking
Run the command
```
mypy
```

## Formatting
Run the command
```
black --check src
```
