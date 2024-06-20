from fero import Fero
import pandas as pd
from fero.analysis import (
    CombinationConstraint,
    CombinationConstraintOperandType as operands,
    CombinationConstraintOperator as operators,
)

# import time
import os

client = Fero(
    "https://app-stg.ferolabs.com/",
    username="ted-test",
    password=os.environ["PW"],
    verify=True,
)
# live_ds_uuid = "f377ee3d-1ef2-443c-a48f-28fe5dece062"  # stage
# ds = client.get_datasource(live_ds_uuid)
# ds.append_csv("../../data/flexible_opts/live_opt_points/MISSING_KEY.csv")

# live_ds_uuid = "cb2582f3-6b30-4340-8283-e20ec25cf80c" # stage other
# live_ds_uuid = "12506cf5-f02f-4c02-bda3-6f41c8736591"  # local for flex!
# live_ds_uuid = "1ee82567-c106-4be3-8116-3592d3156245"  # fixture
# live_ds_uuid = "a8ec4686-f20a-41ee-9f4b-1529a347d96e"  # SteelQuality with holes

# client = Fero(
#     "http://localhost:8080/",
#     username="live_prediction_bot",
#     password="C4TMTkyhF",
#     verify=True,
# )

# client = Fero(
#     "http://localhost:8080/", username="fero", password="password", verify=True
# )

# process_uuid = "e33d9a3e-ab4d-4d44-8b50-45168cf671c7"
# process = client.get_process(process_uuid)
# data = process.get_data(tags=process.tags, include_batch_id=True)
# print(data.index)
# print(data, flush=True)
# print(">>>?????", flush=True)
# batch_pims_uuid = "47afa600-63b6-4cab-a98a-384727b81803"
# adv_uuid = "d9fd5cf8-8ef9-4836-9d9d-98746e600552"
# process = client.get_process("690eed22-3633-4871-b99d-dcf07156c0f4")
# a_uuid = "7e47394c-36c1-4cbc-8367-ea0a0990eff4"
# analysis = client.get_analysis(a_uuid)
# factors = analysis.factor_names
# targets = analysis.target_names
# print(factors)
# print(targets)
# df = process.get_data(tags=process.tags)

# print(df)

# client = Fero("http://localhost:8080/", username="v1_user", password="password", verify=True)
# live_ds_uuid = "25ed9459-af92-40db-9fb7-97fbcd642db2" # local
# live_ds_uuid = "0fe5987e-248c-4982-aefb-f0f17a735750" # new live
# live_ds_uuid = "5a8be617-5d54-4687-924b-87155b0a28a2"

# ds = client.get_datasource(live_ds_uuid)
# ds.append_csv("../../data/fixture_explainit/upload.csv")
# ds.append_csv("../../data/qa_live_predictions/live_append_missing_cats.csv")


# ds.append_csv("../../data/flexible_opts/fake_data_flex_opt_new.csv")
# ds.append_csv("../../data/flexible_opts/live_opt_points/BID_123.csv")
# print(ds)
# ds.append_csv("../../data/bad_categorical.csv")
# ds.append_csv("../../data/qa_live_predictions/live_append_2.csv")

# revision_model P Only: 351e7c43-3798-49f8-a1d3-5878fb0b754d
# revision_model QQQQ: 25e214b4-8dc1-4175-a159-3abf8e322d6e

# # live_ds = "ad7bb52c-4b88-471b-9a11-90fdb6157d5e" # <- local
# # live_ds = "757e4c2c-4324-4c43-93f9-cd9edd84a67e"
# # live_ds = "389a7744-0b8f-45bc-87cc-190de0d9afa7" # <- stage_original
# # live_ds = "a26c1e9c-fbd6-4dae-87bc-4d410beae7ab" # <- stage
# # live_ds = "114c9f81-08c0-4aa5-b394-9e5069d56b23" #<- live stage for smoothing
# # same_pk_live = "22defed5-b037-4bdc-8518-ee9675d344b9"
# # diff_pk_live = "5c9f8e63-ca41-40fa-be25-72586fa8de9d"
# # same_pk_ds = client.get_datasource(same_pk_live)
# # same_pk_ds.append_csv("live_test_demo/v2_append_SAME_PK_NAME.csv")
# # diff_pk_ds = client.get_datasource(diff_pk_live)
# # diff_pk_ds.append_csv("live_test_demo/v2_append_DIFFERENT_PK.csv")

# # datasource = client.get_datasource(live_ds)
# # datasource.append_csv("../../data/FF_bad_types.csv")

# # df = pd.read_csv("../../data/3_res_no_kpi_TODAY.csv")
# # df['dt'] = pd.to_datetime(df['dt'])
# # df.set_index("Unnamed: 0", inplace=True)
# # n = 11
# # size = 5
# # row = df.loc[n - 1:n+size,:]
# # delta = pd.Timestamp.utcnow() - max(row['dt'])
# # row.dt = row.dt + delta #[pd.Timestamp.utcnow() - pd.Timedelta(seconds=0.1*i)for i in range(size+1)]
# # print(row)
# # row.to_csv("../sandbox/update_compressed_{}.csv".format(n))

# # datasource = client.get_datasource(live_ds)
# # blah = pd.read_csv("../sandbox/update_compressed_{}.csv".format(n))
# # datasource.append_csv("../sandbox/update_compressed_{}.csv".format(n))
# # print(blah)
# # datasource.append_csv("../../data/3_res_no_kpi_TODAY.csv") # ../sandbox/mccain_live/opt_row_{}.csv".format(n))


# # client = Fero("http://localhost:8080/", username="fero", password="password", verify=True)
# # ds = client.get_datasource('424d1339-d5a7-43d4-a119-48948d348c94')
# # ds.append_csv("../../data/three_batches_str_ids_AUX_UNION.csv")
# analysis_uuid = "0367fadd-131b-4386-b794-f2aac4fc9533" # local V1
analysis_uuid = "b1af9572-1dec-4cd0-8704-1d9c231b8dd2"  # local v2
# analysis_uuid = "ebb4b669-92bf-4c7b-be23-720f738df271" # fault
analysis = client.get_analysis(analysis_uuid)
# # print(analysis)
# # print(analysis)
# # print(f"Analysis schema: {analysis._schema}")
# df = pd.read_csv("../../data/SteelQuality_Categories_Holes_Int_Target.csv")
# # df = pd.read_csv("../../data/Surface_Defect_Data_with_Categories.csv")
# # df = df.fillna(0.0)
# # d = df.iloc[1].to_dict()

# predict = analysis.make_prediction(df)
# print(predict)


# row = df.tail(50)
# # # print(row)
# # # row.to_csv("TEMP_AYO_INPUT.csv")
# # predict = analysis.make_prediction(row)
# # print(predict)
# # predict.to_csv("TEMP_AYO_PREDICTION.csv")

goal = {
    "goal": "minimize",
    "type": "COST",
    "cost_function": [
        {"name": "Int 2", "min": 5, "max": 24, "cost": 30.0},
        {"name": "Lookup Int 1", "min": 2.0, "max": 8.0, "cost": 50.0},
    ],
}

constraints = [
    {"name": "Target 1", "min": 100.0, "max": 300.0},
]

fixed_factors = {
    "Int 1": 61,
    "Int 3": 26,
    "Join Key": "F",
    "Other": 21,
    "Float 2": 57.33658,
    "Float 3": 196.542,
    "T1 LB": 109.0425,
    "T1 UB": 327.127,
}

combination_constraints = {
    "conditions": [
        {
            "kind": "condition",
            "operation": {
                "operand": {"kind": "formula", "value": "'Int 2' / 'Int 2' + 3"},
                "operator": ">",
            },
            "target": {"kind": "formula", "value": "'Int 2' / 'Lookup Int 1'"},
        },
        {
            "kind": "condition",
            "operation": {
                "operand": {"kind": "constant", "value": 131.0},
                "operator": "<",
            },
            "target": {"kind": "formula", "value": "'Target 1 (Mean)' + 1"},
        },
        {
            "kind": "condition",
            "operation": {
                "operand": {"kind": "column", "value": "Other"},
                "operator": ">",
            },
            "target": {"kind": "formula", "value": "abs('Int 2' + 'Other')"},
        },
    ],
    "join": "AND",
    "kind": "clause",
}
combination_constraints = [
    CombinationConstraint(
        ("'Int 2' / 'Lookup Int 1'", operands.FORMULA),
        operators.LESS_THAN,
        (3, operands.CONSTANT),
    ),
    CombinationConstraint(
        ("'Target 1 (Mean)' + 1", operands.FORMULA),
        operators.LESS_THAN,
        (131.0, operands.CONSTANT),
    ),
    CombinationConstraint(
        ("abs('Int 2' + 'Other')", operands.FORMULA),
        operators.GREATER_THAN,
        ("Other", operands.COLUMN),
    ),
]


opt = analysis.make_optimization(
    "Client Opt",
    goal,
    constraints,
    fixed_factors,
    include_confidence_intervals=True,
    combination_constraints=combination_constraints,
)
my_results = opt.get_results()
print(my_results)

# #    "Chemical 2": 106.32,
# #    "Chemical 5": 2,


# cols = ['Chemical 12', 'Chemical 2', 'TensileStrength (5%)', 'TensileStrength (25%)', 'TensileStrength (Mean)', 'TensileStrength (75%)', 'TensileStrength (95%)']
# for col in cols:
#   print("{} -> {}".format(col, my_results[col]))

# fero_client = Fero(username='whi_ops_worker', password='QJHCJVRZ')
# analysis = fero_client.get_analysis('fec24dca-1243-4862-92d5-e5e1d4a75500')

# goal = {
#   "goal": "minimize",
#   "type": "COST",
#   "cost_function": [
#       {"name": "MN", "min": 0.5, "max": 1.35, "cost": 4101.62},
#       {"name": "V", "min": 0.0, "max": 0.15, "cost": 41647.2}
#   ]
# }

# print(f"Analysis schema: {analysis._schema}")

# constraints = [{"name": "TS", "min": 71000.0, "max": 95000.0},
#                {"name": "YS", "min": 56000.0, "max": 70000.0}]

# opt = analysis.make_optimization("sample_cost_optimization", goal, constraints)
# print(opt.get_results())


# fero_client = Fero(username='brian_for_testing', password='HRFIXJQF')
# process = fero_client.get_process('f923bf4d-c773-4f53-8232-ceeebbb7c7b3')
# # print(process.tags)
# df_process_data = process.get_data(process.tags,
#                                    kpis = ['644_Y_197_164.Value'])
# print(df_process_data['Timestamp'].min())
# print(df_process_data['Timestamp'].max())
