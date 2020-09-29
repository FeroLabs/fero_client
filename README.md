# fero_client

`fero` is a client side python library intended to help users interact with the Fero Machine Learning website. The primary entrypoint for interacting with the Fero Application is via the `Fero` client class

## Quickstart

```python
from fero import Fero

# create a client
fero_client = Fero()

# get an analysis
analyses = fero_client.search_analyses(name="quality-example-data")

analysis = analyses[0]

# Create a data frame with the expected input columns
df = pd.DataFrame([{"value": 5, "value2": 2}])

# Make a prediction
prediction = analysis.make_prediction(df)

print(prediction)
'''
      value	  value2	target_high	 target_low	  target_mid
0	  5	      2	        100	         75	          88
'''
```

## Providing Credentials

The simplest way to provide your Fero login credentials to the API is as arguments to the `Fero` object on initialization.

```python
fero_client = Fero(username="<your username>", password="<your password>")
```

However, while this is fine for interactive shells it's not ideal for a publicly viewable script. To account for this `Fero` also supports setting the `FERO_USERNAME` and `FERO_PASSWORD` environment variables or storing your username and password in a `.fero` in the home directory. This file needs to be in the the following format.

```
FERO_USERNAME=fero_user
FERO_PASSWORD=shouldBeAGoodPassword
```

If you're using the `Fero` client to access an on premise installation, both the hostname for the local Fero site can be provided with `hostname="https://local.fero-site"` and an internal CA via `verify="path/to/ca-bundle`. Verify is passed directly to requests so can be completely disabled by passing `False`

```python
local_client = Fero(hostname="https://fero.self.signed", verify=False)
```

## Finding An Analysis

An `Analysis` object is the how Fero exposes ML models to the API. The Fero client provides two different methods to find an `Analysis`. The first is `Fero.get_analysis` which takes a single UUID string and attempts to lookup the analysis by its unique id. The second method is `Fero.search_analyses` which will return a list of available `Analysis` objects. If no keyword arguments are provided, it will return all analyses you have available on the Fero website. Optionally, `name` can be provided to filter to only analyses matching that name.

#### Examples

```python
from fero import Fero
fero_client = Fero()

# get a specific analysis
analysis = fero_client.get_analysis('5dfbbb63-8ad4-4638-9fdb-61e39952d3cf')

# get all available analyses
all_analyses = fero.search_analyses()

# only get "quality" analyses
quality_only =  fero.search_analyses(name="quality")
```

## Using an Analysis

Along with associated properties such as `name` and `uuid`, an `Analysis` provides two methods for interacting with Fero.

The first is `Analysis.has_trained_model` which is simply a boolean check that a model has finished training. This will be false if the Analysis is still training or there was an error training and it has not been revised.

The second method is `Analysis.make_prediction` which, as its name implies, makes a prediction using the latest model associated with the analysis. This function can take either a pandas data frame with columns matching the expected inputs(factors) for the model or a list of dictionaries with each dictionary containing a key/value pairs for each factor. A prediction will be made for each row in the data frame or each dictionary in the list.

The return value will either be a data frame with each target value predicted by Fero added to each row or keys for each target added to each dictionary depending on the initial input type. These values will have the suffixes `_low`, `_mid`, `_high` added to each target name to indicate the range of the prediction.

### Example

```python
raw_data = [{"value": 5, "value2": 2}]

# Using a data frame
df = pd.DataFrame([raw_data])
prediction = analysis.make_prediction(df)

print(prediction)
'''
      value	  value2	target_high	 target_low	  target_mid
0	  5	      2	        100	         75	          88
'''

# Using a list of dicts
print(analysis.make_prediction(raw_data))
'''
[{"value": 5, "value2": 2, "target_high": 100, "target_low": 75, "target_mid": 88}]
'''
```
