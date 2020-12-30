import streamlit as st  # type:ignore
import pandas as pd  # type:ignore
import plotly.express as px


st.sidebar.markdown("""# Hyperparameter tuning benchmark""")

datasets = pd.read_csv("./data/openml_datasets.csv")
results = pd.read_csv("./data/openml_results.csv")

st.write(datasets)
st.write(results)


# best_models["max_features"] = best_models["max_features"].apply(lambda x: round(x, 4))
# best_models["max_samples"] = best_models["max_samples"].apply(lambda x: round(x, 4))

results = results.drop(["name", "improvement", "total_data_points", "samples", "features", "classes"], axis=1)


fig = px.parallel_coordinates(
    results,
    color="test_score_tuned",
    # labels={
    #     "": "",
    #     "": "",
    #     "": "",
    #     "": "",
    #     "": "",
    # },
    color_continuous_scale=px.colors.diverging.Tealrose,
    color_continuous_midpoint=0.5,
)
fig.show()
