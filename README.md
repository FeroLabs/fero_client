# fero_client

`fero` is a client-side Python library intended to help users interact with [Fero](https://app.ferolabs.com). 

## Quickstart

```python
from fero import Fero

# Create a Fero client object
fero_client = Fero()

# Get a specific analysis by its unique identifier
analysis = fero_client.get_analysis("5dfbbb63-8ad4-4638-9fdb-61e39952d3cf")

# Create a pandas DataFrame with factor values for this analysis
df = pd.DataFrame([{"value": 5, "value2": 2}])

# Make a prediction
prediction = analysis.make_prediction(df)

print(prediction)
'''
   value   value2  target_low90  target_low50 target_mid target_high50  target_high90
0      5        2            70            75         80            88             92
'''
```

## Providing Credentials

The simplest way to provide your Fero login credentials is as arguments to the `Fero` object on initialization.

```python
fero_client = Fero(username="<your username>", password="<your password>")
```

While this is fine for interactive shells, it is not ideal for a publicly viewable script. To account for this, `Fero` also supports setting the `FERO_USERNAME` and `FERO_PASSWORD` environment variables or storing your username and password in a `.fero` file in the home directory. This file needs to be in the the following format.

```
FERO_USERNAME=fero_user
FERO_PASSWORD=shouldBeAGoodPassword
```

If you are using the `Fero` client to access an on-premises installation, both the hostname for the local Fero server can be provided with `hostname="https://local.fero-site"` and an internal SSL certification via `verify="path/to/ca-bundle`. (See [here](https://docs.python-requests.org/en/master/user/advanced/#ssl-cert-verification) for additional details.) Verify is passed directly to the underlying Python `requests` package; thus, if you desire, verification can be disabled by passing `verify=False`.

```python
local_client = Fero(hostname="https://fero.self.signed", verify=False)
```

## Finding a Fero Analysis

The Fero client provides two different methods to find an `Analysis`. The first is `Fero.get_analysis` which takes a single unique identifier string (UUID) and attempts to look up the analysis matching this ID. The second method is `Fero.search_analyses` which will return an iterator of available `Analysis` objects. If no keyword arguments are provided, it will return all analyses you have available on Fero. Optionally, `name` can be provided to filter to only analyses matching that name.

#### Examples

```python
from fero import Fero
fero_client = Fero()

# Get a specific analysis
analysis = fero_client.get_analysis("5dfbbb63-8ad4-4638-9fdb-61e39952d3cf")

# Get all available analyses
all_analyses = fero_client.search_analyses()

# Only get "plant_A" analyses
plant_A_only =  fero_client.search_analyses(name="plant_A")
```

## Using an Analysis

Along with associated properties such as `name` and `uuid`, an `Analysis` provides a variety of methods for interacting with Fero.

The first thing to call when working with an Analysis is `Analysis.has_trained_model`, which checks whether the Analysis is ready to use. This will be false if the Analysis is still being configured or if there was an error during configuration. 

### Making a simple prediction

The `Analysis.make_prediction` method makes a prediction using the latest revision of the Analysis. This function can take either a pandas `DataFrame` with columns matching the expected factors or a list of dictionaries with each dictionary containing a key/value pairs for each factor. A prediction will be made for each row in the `DataFrame` or each dictionary in the list.

The return value will either be a `DataFrame` or a dictionary, depending on the initial input type. These values will have the suffixes `_lowX`, `_mid`, `_highX` added to each target name to indicate the prediction intervals. Specifically:
- `target_low90` corresponds to the 5% prediction level,
- `target_low50` corresponds to the 25% prediction level,
- `target_mid` corresponds to the mean prediction,
- `target_high50` corresponds to the 75% prediction level, and
- `target_high90` corresponds to the 95% prediction level.

The naming convention indicates that:
- 50% of the time, the corresponding measurement should fall between `(target_low50, target_high50)`, and
- 90% of the time, the corresponding measurement should fall between `(target_low90, target_high90)`.

#### Example

```python
raw_data = [{"value": 5, "value2": 2}]

# Using a DataFrame
df = pd.DataFrame([raw_data])
prediction = analysis.make_prediction(df)

print(prediction)
'''
   value   value2  target_low90  target_low50 target_mid target_high50  target_high90
0      5        2            10            20         30            40             50
'''

# Using a list of dicts
prediction = analysis.make_prediction(raw_data)
print(prediction)
'''
[{"value": 5, "value2": 2, "target_low90": 10, "target_low": 20, "target_mid": 30, "target_high50": 40, "target_high90": 50}]
'''
```

### Optimize

A more advanced usage of an `Analysis` is to create an optimization which will make a prediction that satistifies a specified `goal` within the context of `constraints` on other factors or targets. For example, if you wanted to the minimize value of `value` while keeping `target` within a set range, you would provide the following goal and constraint configurations.

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

Fero also supports the idea of a cost optimization, which will weight different factors by specified cost multipliers to find the best combination of inputs. For example, to find the minimum combined cost of `value` and `value2` while meeting the expected values of `target`, you could do the following:

```python

goal = {
  "goal": "minimize",
  "type": "cost",
  "cost_function": [{"name": "value", "min": 50.0, "max": 100.0, "cost": 5.0}, {"name": "value2", "min": 70.0, "max": 80.0.0, "cost": 9.0}]
}

constraints = [{"name": "target", "min": 100.0, "max": 200}]

opt = analysis.make_optimization("example_cost_optimization", goal, constraints)
```

In both cases, a `Prediction` object is returned, which will provide access to the results of the optimization. By default, the result will be a `DataFrame` but it can also be configured to be a list of dictionaries by specifying `format="record"` in `get_results`.

#### Example

```python

goal = {
  "goal": "minimize",
  "factor": {"name": "value", "min": 50.0, "max": 100.0}
}

constraints = [{"name": "target", "min": 100.0, "max": 200}]

opt = analysis.make_optimization("example_optimization", goal, constraints)

print(opt.get_results())
'''
   value   value2   target (5%)   target (Mean)   target (95%)
0     60       40           100             150            175
'''
```

## Finding a Fero Asset

The Fero client provides two different methods to find an `Asset`. The first is `Fero.get_asset`, which takes a single unique identifier string (UUID) and attempts to look up the asset matching this ID. The second method is `Fero.search_assets`, which will return an iterator of available `Asset` objects. If no keyword arguments are provided, it will return all assets you have available on the Fero website. Optionally, `name` can be provided to filter to only assets matching that name.

#### Examples

```python
from fero import Fero
fero_client = Fero()

# Get a specific asset
asset = fero_client.get_asset("fd57ba36-3c5d-40f5-ae0c-d7b76ab39ee5")

# Get all available assets
all_assets = fero_client.search_assets()

# Get only "plant_B" assets
plant_B_only = fero_client.search_assets(name="plant_B")
```

## Using an Asset

Along with associated properties such as `name` and `uuid`, an `Asset` provides a few methods for interacting with Fero.

The first thing to call when working with an asset is `Asset.has_trained_model`, which checks whether the Asset is ready to use. This will be false if the Asset is still being configured or if there was an error during configuration.

### Making a prediction

The `Asset.predict` method makes a prediction using the latest revision of the Asset. Fero computes predictions for all controllable factors and with those results, predictions for all target variables. Predictions are provided for the 5 time intervals following the end of the training dataset. (Interval size is determined during model configuration and training.) Optionally, you may call `Asset.predict` with an argument specifying values for one or more of the controllable factors; Fero will predict all targets using your specified values in place of its controllable factor predictions where applicable.

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

# Provide specified values as a DataFrame
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

print(prediction["mean:Factor2"])
'''
[
    13.452, 13.119, 13.084, 13.003, 12.976
]
'''

print(prediction["index"])
'''
[
    2020-12-25T00:00:00Z, 2020-12-25T01:00:00Z, 2020-12-25T02:00:00Z, 2020-12-25T03:00:00Z, 2020-12-25T04:00:00Z
]
'''
```

## Fero Processes

The Fero client provides two different methods to find a `Process`. The first is `Fero.get_process` which takes a single unique identifier string (UUID) and attempts to look up the analysis matching this ID. The second method is `Fero.search_processes` which will return an iterator of available `Process` objects. If no keyword arguments are provided, it will return all processes you have available on Fero. Optionally, `name` can be provided to filter to only processes matching that name.

Processes represent data via two main underlying entities, the `Tag` and the `Stage`. A `Tag` is a column of a specific measurement in the underlying data. A `Stage` is a logical part of a process consisting of various tags and an order relative to the other stages. For example, a steel process might have first stage for melting the steel and a later stage for casting the steel, each with corresponding measurements in the form of tags.

### Example

```python

from fero import Fero
fero_client = Fero()

# Get a single process
process = fero_client.get_process("c6f69e96-db4d-43ed-8837-d5827cc81112")

# Search processes by name
processes = [p for p in fero_client.search_processes(name="process X")]

# Get the tags of the process
tags = process.tags

# Get stages of the process
stages = process.stages
```

## Downloading Process Data

A `Process` object can be used to download the pandas `DataFrame` that the process would produce for analysis. Because not all tags are generally used in a analysis, a list of desired tags is required before data can be downloaded. Additionally, a target or key performance indicator (kpi) tag can be set while requesting data. Functionally, this will limit the data returned to the stage of the kpi tag and any preceding stages. For advanced and batch processes, kpis are optional; however, they are required for continuous processes because the data is computed using the observed times of the kpi.

### Example

```python

# Get all data for single process
process = fero_client.get_process("9777bae7-95af-4bea-98b9-c703ab940a05")

df = process.get_data(process.tags)

print(df)
'''
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
'''

# Limit the process to an earlier kpi

df = process.get_data(["s1_factor1", "s3_kpi"], kpis=["s2_factor1"])

'''
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
'''

```
