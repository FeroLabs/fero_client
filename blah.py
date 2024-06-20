import numpy as np
import pandas as pd
from typing import Optional, Any
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_object_dtype
from fero import Fero

username = "ted"
password = "Pizza 3.14"

user = "2021-07-14_testing corp_admin"
client = Fero(
    username=username, password=password, hostname="https://app-stg.ferolabs.com/"
)
client.impersonate(user)
process_id = "919bc891-34b1-463d-970a-7b6dddc29d2c"
analysis_id = "d18e6e37-de68-4472-9157-f99fb3d71133"

analysis = client.get_analysis(uuid=analysis_id)
process = client.get_process(uuid=process_id)

tags = process.tags
target_labels = analysis.target_names
data = process.get_data(
    tags=tags,
    kpis=target_labels,
)

factor_labels = analysis.factor_names
optimize_factors = data.iloc[2][factor_labels[:3]].to_dict()
fixed_factors = data.iloc[2][factor_labels[3:]].to_dict()

goal = {
    "goal": "maximize",
    "factor": {
        "name": target_labels[0],
        "min": data[target_labels[0]].min(),
        "max": data[target_labels[0]].max(),
    },
}

constraints = []
for f in optimize_factors:
    lower_bound = data[f].min()
    upper_bound = data[f].max()
    constraints.append(
        {
            "name": f,
            "min": lower_bound,
            "max": upper_bound,
        }
    )


opt = analysis.make_optimization(
    "example_target_optimization",
    goal,
    constraints,
    fixed_factors=fixed_factors,  # this causes an error
)

print(opt.get_results())
