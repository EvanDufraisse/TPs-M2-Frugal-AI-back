
# practical - SHORT_



## Installation

Create an environment with the following command:

```bash
conda create -n <name_of_your_choice> python=3.10 %version of python you want to use
```
Then activate
```bash
conda activate <name_of_your_choice>
```

Then install the package with the following command in the root folder of the project:

```bash
pip install -e ./
# -e stems for editable, meaning that you can modify the code and the changes will be taken into account dynamically
```

## Usage

#### CLI usage

```bash
python ./src/practical/main.py "John Doe"
```
prints "Hello John Doe!"

#### Python usage

```python
import practical
# or
from practical.backend.hello import print_hello_name

## Miscellaneous

This repo also contains a handy script to convert jupyter notebooks to easy versionnable versions.
Jupyter notebooks are a pain with git. The script convert_jupyter.py will duplicate and remove all cells outputs of jupyter notebooks to make them versionable.

- You need to install pandoc
- You need to install nbconvert
