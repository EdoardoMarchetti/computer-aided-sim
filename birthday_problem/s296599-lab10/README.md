# Birthday problem simulator

To run the simulation, firstly recreate my virtual environment with the command

```pip install -r requirements.txt```

Secondly, take a look at the input parameters typing

```python processor.py --help```

and then execute accordingly.

Note that with "first problem" we intend the computation of the
average size of the population to have at least one collision,
while with "second problem" we mean the computation of the collision
probability against an arbitrary number of m values (population size).

To obtain the exact same results of the report please use the following command.

```python processor.py --k1 100000 --k2 1000 --start 2 --stop 100 --step 1 --distribution realistic --confidence 0.99 --seed 42```

# Data

All the file related to birthday data are stored in the data folder.

For estimating the realistic distribution of birthdays, we used data from the website <a href="https://github.com/fivethirtyeight/data/tree/master/births">fivethirtyeight</a>.

The raw data is stored in the file

```birth_data.csv```

To obtain the random variable estimation you should run the script

```python preprocess.py```

while to plot the CDF you type

```python print_real_distr.py```
