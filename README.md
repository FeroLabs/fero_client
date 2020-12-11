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

Along with associated properties such as `name` and `uuid`, an `Analysis` provides a variety of methods for interacting with Fero.

The first thing to call when working with an analysis is `Analysis.has_trained_model` which is simply a boolean check that a model has finished training. This will be false if the Analysis is still training or there was an error training and it has not been revised. Once you have a model trained you then begin working with the analysis to leverage the model.

### Making a simple prediction

The `Analysis.make_prediction` method, as its name implies, makes a prediction using the latest model associated with the analysis. This function can take either a pandas data frame with columns matching the expected inputs(factors) for the model or a list of dictionaries with each dictionary containing a key/value pairs for each factor. A prediction will be made for each row in the data frame or each dictionary in the list.

The return value will either be a data frame with each target value predicted by Fero added to each row or keys for each target added to each dictionary depending on the initial input type. These values will have the suffixes `_low`, `_mid`, `_high` added to each target name to indicate the range of the prediction.

Note: `Analysis.make_prediction` utilizes a fast bulk prediction operation supported by many Fero models. For some legacy blueprints, you may use `Analysis.make_prediction_serial` to fetch predictions one row at a time.

#### Example

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

### Optimize

A more advanced usage of an `Analysis` is to create an optimization which will make a prediction that satistifies a specified `goal` within the context of `constraints` on other factors or targets. For example, if you wanted to the minimum value of `value` while keeping `target` within a set constraints you would provide the following goal and constraint configurations.

```python

goal = {
  "goal": "minimize",
  "factor": {"name": "value", "min": 50.0, "max": 100.0}
}

constraints = [{"name": "target", "min": 100.0, "max": 200}]

opt = analysis.make("example_optimization", goal, constraints)
```

By default, Fero will use the median values of fixed factors while computing the optimization. These can be overridden with custom values by passing a dictionary of `factor`:`value` pairs as the `fixed_factors` argument to the optimization function.

Fero also supports the idea of a cost optimization which will weight different factors by a cost multiplier to find the best combination of inputs. For example, to find the minimum cost between `value` and `value2` while meeting the expected values of `target` you could do the following

```python

goal = {
  "goal": "minimize",
  "type": "cost
  "cost_function": [{"name": "value", "min": 50.0, "max": 100.0, "cost": 5.0}, {"name": "value2", "min": 70.0, "max": 80.0.0, "cost": 9.0}]
}

constraints = [{"name": "target", "min": 100.0, "max": 200}]

opt = analysis.make("example_cost_optimization", goal, constraints)
```

In both cases, a `Prediction` object is return which will provide the data. By default, the result will be a `DataFrame` but it can also be configured to be a list of dictionaries if you're trying to avoid pandas.

#### Example

```python

goal = {
  "goal": "minimize",
  "factor": {"name": "value", "min": 50.0, "max": 100.0}
}

constraints = [{"name": "target", "min": 100.0, "max": 200}]

opt = analysis.make("example_optimization", goal, constraints)

print(opt.get_results())
"""
      value	  value2	 target (5%)  target (Mean)	  target (95%)
0	  60	     40	        100	         150             175
"""
```
