import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np

from dash.dependencies import Input, Output
from plotly import graph_objs as go
from plotly.graph_objs import *
from datetime import datetime as dt

####### modificaciones #######
import environs
import json
import requests
from dash_table import DataTable


# enviroment variables
env = environs.Env()
env.read_env()
API_HOST = env.str("API_HOST")
API_PORT = env.str("API_PORT")

url = f'{API_HOST}:{API_PORT}/api/grouping/'

dropdown_options = {
    'categories': json.loads(requests.post(f'{API_HOST}:{API_PORT}/api/grouping/list/').text)
}

# table headers
columns = [
    {"name": "Category","id": "Category"},
    {"name": "Subcategory", "id": "Subcategory"},
]

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server

# Layout of Dash App
app.layout = html.Div(
    children=[
        html.Div(
            className="row",
            children=[
                # Column for user controls
                html.Div(
                    className="four columns div-user-controls",
                    children=[
                        html.Img(
                            className="logo", src=app.get_asset_url("dash-logo-new.png")
                        ),
                        html.H2("Timeseries - ranking BETA"),
                        html.P(
                            """Select different days using the date picker or by selecting 
                            different time frames on the histogram."""
                        ),
                        html.Div(
                            className="div-for-dropdown",
                            children=[
                                dcc.DatePickerSingle(
                                    id="date-picker",
                                    min_date_allowed=dt(2014, 4, 1),
                                    max_date_allowed=dt(2014, 9, 30),
                                    initial_visible_month=dt(2014, 4, 1),
                                    date=dt(2014, 4, 1).date(),
                                    display_format="MMMM D, YYYY",
                                    style={"border": "0px solid black"}
                                )
                            ],
                        ),
                        # Change to side-by-side for mobile layout
                        html.Div(
                            className="row",
                            children=[
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        # Dropdown to select times
                                        dcc.Dropdown(
                                            id='dropdown',
                                            options=[{'label': k, 'value': k} for k in dropdown_options['categories']],
                                            placeholder='Select category'
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.P(id="total-rides"),
                        html.P(id="total-rides-selection"),
                        html.P(id="date-value"),
                        html.Button('Update plot', id='updatePlotButton'),
                        dcc.Markdown(
                            children=[
                                "By: [Felipe A.](https://github.com/membrilloski/kubeflow/tree/master/web-ui)"
                            ]
                        ),
                    ],
                ),
                # Column for app graphs and plots
                html.Div(
                    className="eight columns div-for-charts bg-grey",
                    children=[
                        html.Div(
                            className="row",
                            children=[
                                html.Button('Select all', id='selectAllButton'),
                            ]
                        ),
                        
                        DataTable(
                            id='table',
                            columns=columns,
                            data=[],
                            filter_action="native",
                            sort_action="native",
                            sort_mode="multi",
                            row_selectable='multi',
                            page_action="native",
                            page_current= 0,
                            page_size= 6,
                            style_as_list_view=True,
                            style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                            style_cell={
                                'backgroundColor': 'rgb(50, 50, 50)',
                                'color': 'white'
                            },
                        ),
                        html.Div(
                            className="text-padding",
                            children=[
                                "Select the category and data range by using the date picker to get the predictions"
                            ],
                        ),
                        dcc.Graph(
                            id='graph',
                            figure={}
                        ),
                    ],
                ),
            ],
        )
    ]
)

@app.callback(
    Output("table", "data"),
    [Input("dropdown", "value")]
)
def updateTable(dropdown_selection):
    if dropdown_selection is None:
        
        return []

    url = f'{API_HOST}:{API_PORT}/api/grouping/'
    data = {'grouping_name': dropdown_selection}
    j_data = json.dumps(data)
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    r = requests.post(url, data=j_data, headers=headers)
    df = pd.DataFrame(json.loads(r.text))
    df = df.rename(columns={"grouping_name": "Category", "name": "Subcategory"})
    plot_data = []
    
    
    return df.to_dict('records')

def get_all_models_data(subcategory_list):
    for subcategory in subcategory_list:
        url = f'{API_HOST}:{API_PORT}/api/model/'
        data = {'model': subcategory, 'future_periods': 1, 'freq': 'D'}
        j_data = json.dumps(data)
        headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        r = requests.post(url, data=j_data, headers=headers)
        df = pd.DataFrame(json.loads(r.text))
        plot_data = generate_plot_data_from_df(df,'ds','yhat',name=subcategory)
        yield plot_data


def generate_plot_from_list(plot_list):
    """
    Get plot data from every item contained in the list
    """
    plot_data = []
    for subcategory in get_all_models_data(plot_list):
        plot_data.append(subcategory)

    figure = {'data': plot_data, 'layout': {'clickmode': 'event+select'}}

    return figure

def generate_plot_data_from_df(df,x,y,name='Unnamed',mode='lines', marker={'size': 12}):
    data = {
        'x': list(df[x].values),
        'y': list(df[y].values),
        'name': name,
        'mode': mode,
        'marker': marker
    }
    return data

@app.callback(
    Output('graph','figure'),
    [dash.dependencies.Input('updatePlotButton', 'n_clicks'),
    dash.dependencies.Input('table', 'data'),
    dash.dependencies.Input('table', 'selected_rows')])
def update_output(n_clicks, rows, selected_rows):
    if n_clicks is None:
        return {}
    else:
        selected_rows_names=[rows[i]['Subcategory'] for i in selected_rows]
        figure = generate_plot_from_list(selected_rows_names)
        return figure

if __name__ == "__main__":
    app.run_server(debug=True)
