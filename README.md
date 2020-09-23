# fero_client

`fero` is a client side python library intended to help users interact with the Fero Machine Learning application. The primary entrypoint for interacting with the Fero Application is via the `Fero` client class

## Quickstart

```python
from fero import Fero

# create a client
fero_client = Fero()

# get an analysis
analysis = fero_client.get_analysis("quality-example-data")

data = [
    {"value": 5, "value2": 2}
]

# Make a prediction
prediction = analysis.make_prediction(data)

print(prediction)
'''
[{
    "value": 5,
    "value2": 2,
    "target_high": 100,
    "target_low": 75,
    "target_mid": 88
}]
'''
```
