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

# get an asset
assets = fero_client.search_assets(name="my-favorite-asset")
asset = assets[0]

# see the current predictions
prediction = asset.predict()

print(prediction)
'''
                        mean:Factor1  p5:Factor1  p25:Factor1 ... p75:Target1  p95:Target1
2020-12-25T00:00:00Z    7.937         7.253       7.688       ... 1.921        2.197
2020-12-25T01:00:00Z    8.059         6.962       7.721       ... 1.924        2.202
2020-12-25T02:00:00Z    8.193         6.754       7.692       ... 1.871        2.318
2020-12-25T03:00:00Z    8.349         6.552       7.619       ... 1.830        2.375
2020-12-25T04:00:00Z    8.492         6.199       7.498       ... 1.762        2.425
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

An `Analysis` object is the how Fero exposes ML models to the API. The Fero client provides two different methods to find an `Analysis`. The first is `Fero.get_analysis` which takes a single UUID string and attempts to lookup the analysis by its unique id. The second method is `Fero.search_analyses` which will return an iterator of available `Analysis` objects. If no keyword arguments are provided, it will return all analyses you have available on the Fero website. Optionally, `name` can be provided to filter to only analyses matching that name.

#### Examples

```python
from fero import Fero
fero_client = Fero()

# get a specific analysis
analysis = fero_client.get_analysis('5dfbbb63-8ad4-4638-9fdb-61e39952d3cf')

# get all available analyses
all_analyses = fero_client.search_analyses()

# only get "quality" analyses
quality_only =  fero_client.search_analyses(name="quality")
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

opt = analysis.make_optimization("example_optimization", goal, constraints)
```

By default, Fero will use the median values of fixed factors while computing the optimization. These can be overridden with custom values by passing a dictionary of `factor`:`value` pairs as the `fixed_factors` argument to the optimization function.

```python

fixed_factors = {
  "value": 10,
  "value2": 20
}

opt = analysis.make_optimization("example_optimization", goal, constraints, fixed_factors)
```

Fero also supports the idea of a cost optimization which will weight different factors by a cost multiplier to find the best combination of inputs. For example, to find the minimum cost between `value` and `value2` while meeting the expected values of `target` you could do the following

```python

goal = {
  "goal": "minimize",
  "type": "cost",
  "cost_function": [{"name": "value", "min": 50.0, "max": 100.0, "cost": 5.0}, {"name": "value2", "min": 70.0, "max": 80.0.0, "cost": 9.0}]
}

constraints = [{"name": "target", "min": 100.0, "max": 200}]

opt = analysis.make_optimization("example_cost_optimization", goal, constraints)
```

In both cases, a `Prediction` object is return which will provide the data. By default, the result will be a `DataFrame` but it can also be configured to be a list of dictionaries if you're trying to avoid pandas.

#### Example

```python

goal = {
  "goal": "minimize",
  "factor": {"name": "value", "min": 50.0, "max": 100.0}
}

constraints = [{"name": "target", "min": 100.0, "max": 200}]

opt = analysis.make_optimization("example_optimization", goal, constraints)

print(opt.get_results())
"""
      value	  value2	 target (5%)  target (Mean)	  target (95%)
0	  60	     40	        100	         150             175
"""
```

## Finding an Asset

An `Asset` object is how Fero exposes ML time series models to the API. The Fero client provides two different methods to find an `Asset`. The first is `Fero.get_asset`, which takes a single string and attempts to lookup the asset by its unique id. The second method is `Fero.search_assets`, which will return an iterator of available `Asset` objects. If no keyword arguments are provided, it will return all assets you have available on the Fero website. Optionally, `name` can be provided to filter to only assets matching that name.

#### Examples

```python
from fero import Fero
fero_client = Fero()

# get a specific asset
asset = fero_client.get_asset('fd57ba36-3c5d-40f5-ae0c-d7b76ab39ee5')

# get all available assets
all_assets = fero_client.search_assets()

# get only "favorite" assets
favorite_only = fero_client.search_assets(name="favorite")
```

## Using an Asset

Along with associated properties such as `name` and `uuid`, an `Asset` provides a few methods for interacting with Fero.

The first thing to call when working with an asset is `Asset.has_trained_model`, which is simply a boolean check that a model has finished training. This will be false if the Asset is still training or there was an error training and it has not been revised. Once you have a model trained you then begin working with the asset to leverage the model.

### Making a prediction

The `Asset.predict` method, as its name implies, makes predictions using the latest model associated with the asset. Fero computes predictions for all controllable factors and with those results, predictions for all target variables. Predictions are provided for the 5 time intervals following the end of the training dataset. (Interval size is determined during model configuration and training.) Optionally, you may call `Asset.predict` with an argument specifying values for one or more of the controllable factors; Fero will predict all targets using your specified values in place of its controllable factor predictions where applicable.

#### Examples

```python
# With no inputs
prediction = asset.predict()

print(prediction.columns)
['mean:Factor1', 'p5:Factor1', 'p25:Factor1', 'p75:Factor1', 'p95:Factor1',
 'mean:Factor2', 'p5:Factor2', 'p25:Factor2', 'p75:Factor2', 'p95:Factor2',
 'mean:Target1', 'p5:Target1', 'p25:Target1', 'p75:Target1', 'p95:Target1']

print(prediction)
'''
                        mean:Factor1  p5:Factor1  p25:Factor1 ... p75:Target1  p95:Target1
2020-12-25T00:00:00Z    7.937         7.253       7.688       ... 1.921        2.197
2020-12-25T01:00:00Z    8.059         6.962       7.721       ... 1.924        2.202
2020-12-25T02:00:00Z    8.193         6.754       7.692       ... 1.871        2.318
2020-12-25T03:00:00Z    8.349         6.552       7.619       ... 1.830        2.375
2020-12-25T04:00:00Z    8.492         6.199       7.498       ... 1.762        2.425
'''

# Provide specified values as a data frame
new_factor_values = pd.DataFrame({
    "Factor1": [8.0, 8.1, 8.2, 8.3, 8.4]
})

prediction = asset.predict(new_factor_values)

print(prediction.columns)
['specified:Factor1',
 'mean:Factor2', 'p5:Factor2', 'p25:Factor2', 'p75:Factor2', 'p95:Factor2',
 'mean:Target1', 'p5:Target1', 'p25:Target1', 'p75:Target1', 'p95:Target1']

print(prediction)
'''
                        specified:Factor1  mean:Factor2  p5:Factor2 ... p75:Target1  p95:Target1
2020-12-25T00:00:00Z    8.0                13.452        11.953     ... 1.921        2.197
2020-12-25T01:00:00Z    8.1                13.119        11.762     ... 1.924        2.202
2020-12-25T02:00:00Z    8.2                13.084        11.454     ... 1.871        2.318
2020-12-25T03:00:00Z    8.3                13.003        11.352     ... 1.830        2.375
2020-12-25T04:00:00Z    8.4                12.976        11.109     ... 1.762        2.425
'''

# Provide specified values as a dictionary
new_factor_values = {
    "Factor1": [8.0, 8.1, 8.2, 8.3, 8.4]
}

prediction = asset.predict(new_factor_values)

print(list(prediction.keys())
'''
[
    'specified:Factor1', 'mean:Factor2', 'p5:Factor2', 'p25:Factor2', 'p75:Factor2',
    'p95:Factor2', 'mean:Target1', 'p5:Target1', 'p25:Target1', 'p75:Target1', 'p95:Target1',
    'index'
]
'''

print(prediction['mean:Factor2'])
'''
[
    13.452, 13.119, 13.084, 13.003, 12.976
]
'''

print(prediction['index'])
'''
[
    2020-12-25T00:00:00Z, 2020-12-25T01:00:00Z, 2020-12-25T02:00:00Z, 2020-12-25T03:00:00Z, 2020-12-25T04:00:00Z
]
'''
```

## Processes

In Fero, a process represents how to combine and transform a variety of raw data sources into a single set of data for analysis based on the physical mechanics of how the industrial process works. Processes can be accessed individually via `api_id` or searched by name using the `Fero` client.

Processes represent data via two main underlying entities, the `Tag` and the `Stage`. `Tags` are columns of a specific measurement in the underlying data. Stages are ordered grouping of various tags that represent logical parts of a process. For example, a steel process might have a stage for melting the steel and a different stage for measurements when casting the steel.

### Example

```python

from fero import Fero
fero_client = Fero()

# get a single process
process = fero_client.get_process('c6f69e96-db4d-43ed-8837-d5827cc81112')

# search processes by name
processes = [p for p in fero_client.search_processes(name="process X")]

# get the tags
tags = process.tags

# get stages
stages = process.stages
```

## Downloading Process Data

`Process` objects can be used to download the pandas data frame that the process would produce for analysis. Because not all tags are generally used in a analysis, a list of tags is required before data can be downloaded. A target or key performance indicator (kpi) is also required for an analysis which can be set while requesting data. Functionally, this will limit that data returned to the stage of the kpi tag and any preceding stages. For advanced and batch processes, kpis are optional, however they are required for continuous processes because the data is computed using the observed times of the kpi.

### Example

```python

# get all data for single process
process = fero_client.get_process("9777bae7-95af-4bea-98b9-c703ab940a05")

df = process.get_data(process.tags)

print(df)
"""
       s1_factor1  s1_factor2  s2_factor1  s3_factor1  s3_factor2   s3_kpi
0               0          14           7        28.5           0     49.5
1               1           8           5        36.0           3     53.0
2               2           2           3        39.5           6     52.5
3               3          10           8        26.5           9     56.5
4               4           4           6        41.5          12     67.5
...           ...         ...         ...         ...         ...      ...
10395       10395           4       10397        38.0       31185  52019.0
10396       10396           0       10396        32.5       31188  52012.5
10397       10397          14       10404        40.5       31191  52046.5
10398       10398           0       10398        25.5       31194  52015.5
10399       10399          14       10406        42.0       31197  52058.0

[10400 rows x 6 columns]
"""

# limit the process to an earlier kpi

df = process.get_data(['s1_factor1', 's3_kpi'], kpis=['s2_factor1'])

"""
                            dt  s2_factor1  s1_factor1
0    2020-03-01 00:00:00+00:00          10        <NA>
1    2020-03-01 00:01:00+00:00         162        <NA>
2    2020-03-01 00:02:00+00:00          12          16
3    2020-03-01 00:03:00+00:00          12          15
4    2020-03-01 00:04:00+00:00          56         162
...                        ...         ...         ...
1994 2020-03-02 09:14:00+00:00        2006          65
1995 2020-03-02 09:15:00+00:00        2007          20
1996 2020-03-02 09:16:00+00:00         415         174
1997 2020-03-02 09:17:00+00:00        2001         166
1998 2020-03-02 09:18:00+00:00           0           2

[1999 rows x 3 columns]
"""

```
