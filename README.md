# recommender-system

```python
import pandas as pd
input_file = '/home/...'

df = pd.read_csv(input_file)
df.head()
```

```python
from recsys.sqlite import *

feedback_to_rating(df = df, db = "db")
create_mapping('db', 'item', sql_query = "SELECT * FROM rating LIMIT 500000")
create_mapping('db', 'user', sql_query = "SELECT * FROM rating LIMIT 500000")
```
