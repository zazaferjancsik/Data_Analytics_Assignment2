import pandas as pd
import numpy as np

# Create random data for the DataFrame
data = np.random.rand(2, 10)  # 2 rows and 10 columns

# Define column names
column_names = [
    "Feature1", "Feature2", "Feature3", "Feature4", "Feature5",
    "Feature6", "Feature7", "Feature8", "Feature9", "Feature10"
]

# Create the DataFrame
df = pd.DataFrame(data, columns=column_names)

# Display the DataFrame
print(df.values[0])
