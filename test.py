from fero import Fero
import pandas as pd

# fero_client = Fero(
#     hostname="https://app-demo.ferolabs.com", username="ted-test", password="Pizza 3.14", verify=False)
# fero_client = Fero(
#    hostname="https://app-stg.ferolabs.com", username="ted-test", password="Pizza 3.14", verify=True
# )
fero_client = Fero(
    hostname="http://localhost:8080", username="test", password="password", verify=False
)


analysis = fero_client.get_analysis("aee536ee-9083-446f-ba93-5c5a17915206")
# analysis = fero_client.get_analysis("28bb4f44-1874-4d6c-883d-b3626094c26a")
# analysis = fero_client.get_analysis("053aabc0-f919-4c27-bff8-29cf2f7b5ddb")
# analysis = fero_client.get_analysis("b772ca1a-be46-4e0f-9d75-edfe086aeeb5")
print(dir(analysis))
# import pdb
# pdb.set_trace()


goal = {
    "goal": "minimize",
    "factor": {"name": "Chemical 2", "min": 155.36, "max": 12.21},
}

constraints = [
    {"name": "TensileStrength", "min": 32.8, "max": 74.5},
    {"name": "Chemical 5", "min": 0.45, "max": 33.55},
    {"name": "Factor 43", "min": 78, "max": 1507},
    {"name": "Yield Strength", "min": 23.0, "max": 61.6},
]

fixed_factors = {
    "Chemical 2": 138.62,
    "Chemical 12": 2.221699999999999,
    "Chemical 1": 17.46,
    "Chemical 5": 1.81,
    "Factor 43": 250,
    "TensileStrength": 58.5,
    "Yield Strength": 46.3,
}
opt = analysis.make_optimization("my_optimization", goal, constraints, fixed_factors)
print(dir(opt))
# import pdb
# pdb.set_trace()
# df = pd.read_csv("../../data/SteelQualityData_with_Categories_HOLES_2.csv")
# df.dropna(inplace=True)
# df_mini = df.head(5)
# df_mini["TensileStrength_low90"]=list(df_mini["Factor 7"])
# print(df_mini.columns)
# ldict = df_mini.to_dict("records")
# res = analysis.make_prediction(df_mini)
# res2 = analysis.make_prediction(df)
# print(res2.columns)
# print(res.columns)
# print(len(df_mini.columns))
# print(len(res.columns))
# print('in res? {}'.format("TensileStrength_low90" in res))
