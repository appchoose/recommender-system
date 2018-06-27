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

```python
conn = sqlite3.connect("db")
df = pd.read_sql_query(con = conn, sql = "SELECT * FROM rating LIMIT 1000000")
user_mapping = pd.read_sql_query(con = conn, sql = "SELECT * FROM user")
item_mapping = pd.read_sql_query(con = conn, sql = "SELECT * FROM item")
conn.close()

df['itemId'] = df['itemId'].map(item_mapping.set_index('item_str')['item_int'].to_dict().get)
df['userId'] = df['userId'].map(user_mapping.set_index('user_str')['user_int'].to_dict().get)
```

```python
from recsys.tfutils import *

write_tf_records(df, by = "itemId", output = "items_for_user")
write_tf_records(df, by = "userId", output = "users_for_item")
```
