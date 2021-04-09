import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import pickle
import json
import dash_table

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

"""load model"""

with open("/Users/Arsal/examples/raltime_anomaly/model_svm.pkl", 'rb+') as f:
    model = pickle.load(f)


"""read_test_data"""
with open("/Users/Arsal/examples/raltime_anomaly/test_df.json", 'r') as myfile:
    data = json.load(myfile)

to= pd.DataFrame.from_dict(data[0].values()).T

prediction = model.predict(to)

"""read_columns"""
with open("/Users/Arsal/examples/raltime_anomaly/model_columns.pkl", 'rb+') as col:
    cols= pickle.load(col)



# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    # dcc.Graph(
    #     id='example-graph',
    #     figure=fig
    # ),
    dcc.ConfirmDialog(id="table_anomaly")
])

app.layout = dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in df.columns],
    data=df.to_dict('records'),
)




if __name__ == '__main__':
    app.run_server(debug=True)