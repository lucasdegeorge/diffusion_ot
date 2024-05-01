# fpdimod


## Description

This python package, named **fpdimod** (**F**okker **P**lanck **DI**ffusion **MO**dels), provides a code to model the distributions, e.g., of
natural images. The code is based on the solver in the low-rank tensor train format with cross approximation approach for solution of the multidimensional Fokker-Planck equation (FPE) [fpcross](https://github.com/AndreiChertkov/fpcross). The package [teneva](https://github.com/AndreiChertkov/teneva) with compact implementation of basic operations in the tensor train format is also used.


## Installation and usage

1. Install [python](https://www.python.org) (version >= 3.8; you may use [anaconda](https://www.anaconda.com) package manager)

2. Install dependencies `pip install POT teneva==0.12.2 fpcross==0.5.1`

3. Run the demo script as `python demo.py`

4. Run the main computations as `python calc.py 2` (2-dimensional problem), `python calc.py 3` (3-dimensional problem) and `python calc.py 7` (7-dimensional problem)
    > The results will be saved into `result` folder

5. Run `python check.py 2`, `python check.py 3` and `python check.py 7` for demonstration of the computation results.


## Authors

**---Anonymized---***
