import pandas as pd
import numpy as np

# create a random dataframe
df = pd.DataFrame(
    data=np.random.randint(low=0, high=100, size=(10, 10)),
    index=["row_{}".format(i) for i in range(10)],
    columns=["col_{}".format(i) for i in range(10)]
)

# save the dataframe to a csv file
df.to_csv("data.csv")