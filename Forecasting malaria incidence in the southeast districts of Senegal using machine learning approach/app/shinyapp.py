import shiny
from shiny import ui, render, reactive
#import shinyswatch
from shinywidgets import output_widget, render_widget
import pandas as pd
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas.tseries.offsets import MonthBegin
import numpy as np
import folium
from io import BytesIO
import base64
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor
import asyncio
import geopandas as gpd
from IPython import display

# Suppress warnings
warnings.filterwarnings('ignore')

# Charger les données et le modèle
def load_data(file_path):
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        df = gpd.read_file(file_path)
    else:
        raise ValueError("Unsupported file format")
    return df

# Charger les données de malaria
df_malaria = load_data(file_path=".../App/data/dfmalaria.xlsx")

# Charger le modèle
xgb_model2 = joblib.load('.../App/model/XGBoostmodel2.pkl')

# Charger les données des régions du Sénégal
senegal_regions = load_data(file_path=".../data/gadm41_SEN_1.json")

# Fonction pour effectuer les prévisions
def predict_future_incidence(model, df, target_label, input_date, forecast_horizon):
    input_date = pd.to_datetime(input_date)
    features = df.drop(columns=['Date', 'DISTRICT', target_label]).columns.tolist()
    data_until_date = df[df['Date'] <= input_date]
    data_for_prediction = data_until_date.groupby('DISTRICT').tail(12)

    predictions_by_district = []
    for district, district_data in data_for_prediction.groupby('DISTRICT'):
        if len(district_data) < 12:
            continue
        district_data = district_data[features]
        dates_predicted = [input_date + MonthBegin(i) for i in range(forecast_horizon)]
        district_prediction_df = pd.DataFrame({'Date': dates_predicted, 'District': district})

        for i in range(forecast_horizon):
            month_features = district_data.iloc[i].values.reshape(1, -1)
            predicted_value = abs(model.predict(month_features)[0])
            district_prediction_df.loc[i, 'Predicted incidence'] = predicted_value

        district_prediction_df['Latitude'] = district_data['latitude'].iloc[0]
        district_prediction_df['Longitude'] = district_data['longitude'].iloc[0]
        predictions_by_district.append(district_prediction_df)

    final_predictions_df = pd.concat(predictions_by_district, ignore_index=True)
    final_predictions_df['Incidence level'] = pd.cut(
        final_predictions_df['Predicted incidence'],
        bins=[float('-inf'), 5, 15, float('inf')],
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )
    final_predictions_df = final_predictions_df[['Date', 'District', 'Predicted incidence', 'Incidence level', 'Latitude', 'Longitude']]
    return final_predictions_df

# Fonction de visualisation des prévisions
def future_predictions12(df):
    df = df.set_index('Date')
    predictions = np.round(df['Predicted incidence'].values, 2)
    predictions = np.nan_to_num(predictions)
    districts = df['District'].unique()

    num_rows = 5
    num_cols = 3

    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=[f'{district}' for district in districts],
        horizontal_spacing=0.1,
        vertical_spacing=0.1
    )

    for i, district in enumerate(districts, start=1):
        district_data = df[df['District'] == district]
        predicted_values_district = predictions[df['District'] == district]

        fig.add_trace(go.Scatter(
            x=district_data.index,
            y=predicted_values_district,
            mode='lines+markers',
            showlegend=False,
            line=dict(color='purple', dash='dash'),
            marker=dict(color='purple', symbol='circle'),
            hoverinfo='x+y',
            hovertemplate="Date: %{x}<br>Incidence: %{y}‰"
        ),
            row=(i - 1) // num_cols + 1,
            col=(i - 1) % num_cols + 1
        )

        if (i - 1) % num_cols == 0:
            fig.update_yaxes(title_text='Incidence (‰)', row=(i - 1) // num_cols + 1, col=(i - 1) % num_cols + 1)

    fig.update_layout(height=900, width=900, showlegend=False)
    return fig

# Fonction pour afficher la carte Folium avec les prédictions et les régions
def plot_prediction_map_with_regions(prediction_result_df, date):
    filtered_df = prediction_result_df[prediction_result_df['Date'] == date]
    m = folium.Map(location=[14.4974, -14.4524], zoom_start=7)

    # Ajouter les régions du Sénégal à la carte
    folium.GeoJson(
        senegal_regions,
        name='Senegal Regions',
        style_function=lambda feature: {
            'fillColor': '#f4f4f4' if feature['properties']['NAME_1'] in ['Kolda', 'Kédougou', 'Tambacounda'] else '#ffffff',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 1,
        },
        highlight_function=lambda feature: {
            'fillColor': '#f4f4f4' if feature['properties']['NAME_1'] in ['Kolda', 'Kédougou', 'Tambacounda'] else '#ffffff',
            'color': 'black',
            'weight': 3,
            'fillOpacity': 1,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['NAME_1'],
            aliases=['Region:'],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
        )
    ).add_to(m)

    incidence_settings = {
        'Low': {'color': '#008000', 'border_color': '#007000', 'radius': 10},
        'Medium': {'color': '#FFFF00', 'border_color': '#FFB900', 'radius': 15},
        'High': {'color': '#FF0000', 'border_color': '#CC0000', 'radius': 17}
    }

    for _, row in filtered_df.iterrows():
        incidence_level = row['Incidence level']
        if incidence_level not in incidence_settings:
            continue
        settings = incidence_settings[incidence_level]

        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=settings['radius'],
            color=settings['border_color'],
            fill=True,
            fill_color=settings['color'],
            fill_opacity=0.5,
            tooltip = (
                f"<b style='font-size:14px;'>District:</b> "
                f"<span style='font-size:14px; font-weight:bold;'>{row['District']}</span><br>"
                f"<b style='font-size:14px;'>Predicted incidence:</b> "
                f"<span style='font-size:14px; font-weight:bold;'>{round(row['Predicted incidence'], 2)}‰</span><br>"
                f"<b style='font-size:14px;'>Incidence level:</b> "
                f"<span style='font-size:14px; font-weight:bold;'>{row['Incidence level']}</span>"
)


        ).add_to(m)

        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            icon=folium.DivIcon(html=f"<div style='font-size: 8pt; margin-left: 10px; white-space: nowrap;'><b>{row['District']}</b></div>")
        ).add_to(m)
    # Ajouter une légende personnalisée
    legend_html = '''
         <div style="position: fixed; 
                     bottom: 50px; left: 10px; width: 150px; height: 120px; 
                     background-color: white; border:2px solid grey; z-index:9999; font-size:14px;">
         <p style="font-weight: bold; text-align: center; margin-top: 5px;">Legend</p>
         <p style="margin: 5px 0;"><span style="background-color: #008000; border-radius: 50%; padding: 5px 10px; margin-right: 10px;"></span>Low incidence</p>
         <p style="margin: 5px 0;"><span style="background-color: #FFFF00; border-radius: 50%; padding: 5px 10px; margin-right: 10px;"></span>Medium incidence</p>
         <p style="margin: 5px 0;"><span style="background-color: #FF0000; border-radius: 50%; padding: 5px 10px; margin-right: 10px;"></span>High incidence</p>
         </div>
         '''
    
    # Ajouter la légende à la carte
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    return m

# Fonction pour intégrer la carte dans l'application Shiny
def render_folium_map(map_object):
    map_data = BytesIO()
    map_object.save(map_data, close_file=False)
    html_map = base64.b64encode(map_data.getvalue()).decode()
    return f'<iframe src="data:text/html;base64,{html_map}" width="100%" height="500"></iframe>'

# Lire le fichier HTML
with open("index.html", "r", encoding="utf-8") as file:
    header_html = file.read()

# Styles CSS pour la barre de navigation et le sidebar
navbar_style = """
<style>
.navbar {
    background-color: green !important; /* Couleur de la barre de navigation */
    padding: 10px 20px;
    font-family: "Montserrat", sans-serif;
    font-size: 16px;
    margin-bottom: 0;
    display: flex;
    justify-content: space-between;
}

.navbar .nav-item {
    margin-right: 10px;
}

.navbar .nav-link {
    color: black;
    text-transform: uppercase;
    transition: background-color 0.3s ease;
    padding: 10px 15px;
    border-radius: 4px;
    display: inline-block;
}

.navbar .nav-link.active,
.navbar .nav-link:hover {
    background-color: #1b9bff !important;
    color: white;
}

.navbar-brand {
    font-weight: bold;
    font-size: 20px;
    color: black;
}

.navbar .nav {
    display: flex;
    align-items: center;
}

.sidebar {
    background-color: #f4f4f4 !important;
    padding: 10px;
    border-radius: 8px;
    border-right: 1px solid #ddd;
    box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1);
    margin-top: 0;
    margin-bottom: 20px;
}



@media (max-width: 768px) {
    .sidebar {
        width: 100%;
        border-right: none;
    }

    .card {
        width: 100%;
    }
}


</style>
"""

# UI de l'application
forecast_ui = ui.page_fluid(
    ui.include_css(Path(__file__).parent / "style.css"),
    ui.HTML(header_html),
    ui.page_sidebar(
        ui.sidebar(
            ui.input_date("date_input", "Start date", value="2021-12-01", format="yyyy-mm-dd"),
            ui.input_slider(id ="horizon", label="Forecast horizon", min=1, max=12, value=6,
                              step=1, ticks=True, sep=''),
            ui.input_action_button("visualize_button", "VISUALIZE"),
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Malaria incidence forecasts by district"),
                ui.output_ui("prediction_plot"),
                full_screen=True
            ),
            col_widths=[12],
            fill=False
        )
    )
)

map_ui = ui.page_fluid(
    ui.include_css(Path(__file__).parent / "style.css"),
    ui.HTML(header_html),
    ui.page_sidebar(
        ui.sidebar(
            ui.output_ui("date_select"),
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Incidence forecast map"),
                ui.output_ui("map_plot"),
                full_screen=True
            ),
            col_widths=[12],
            fill=False
        )
    )
)

data_ui = ui.page_fluid(
    ui.include_css(Path(__file__).parent / "style.css"),
    ui.HTML(header_html),
    ui.page_sidebar(
        ui.sidebar(
             ui.download_button("download_csv", "DOWNLOAD")  # Ajout du bouton
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Predicted Data"),
                ui.output_data_frame("data_grid"),
                full_screen=True
            ),
            col_widths=[12],
            fill=False
        )
    )
)

about_ui = ui.page_fluid(
    ui.include_css(Path(__file__).parent / "style.css"),
    ui.HTML(header_html),
    ui.layout_columns(
        ui.card(
            ui.card_header("About this application"),
            ui.HTML("""
                <p>PrevAPP is an application designed to forecast malaria incidence by district. 
                It uses advanced machine learning models to predict future incidence based on historical data. <br>
                The application provides visualizations and maps to help the National Malaria Control Program optimize the deployment of interventions in South-East Senegal.</p>
                 <p>
                    <strong>Operating instructions :</strong><br><br>
                    1. Select the start date of your forecast.<br>
                    2. Select the number of months to forecast (forecast horizon).<br>
                    3. Click on “Visualize” to obtain incidence forecasts.<br>
                    4. Use the “Map” tab for a monthly view of forecasts.<br>
                    5. Download the data in the "Data" tab.<br>
                    6. Access the application source code in the "Source Code" tab.
                </p>
            """)
       
        ),
        col_widths=[12],
        fill=False
    )
)

code_source_ui = ui.page_fluid(
    ui.include_css(Path(__file__).parent / "style.css"),
    ui.HTML(header_html),
    ui.layout_columns(
        ui.card(
            ui.card_header("Code Source"),
            ui.HTML("""
                <div style="display: flex; justify-content: center; align-items: center; height: 50vh;">
                    <a href="https://github.com/AYLY92/PhD" target="_blank" style="font-size: 24px; color: #1b85ff; text-decoration: none;">
                        👉 Click here to access the source code on GitHub
                    </a>
                </div>
            """),
        ),
        col_widths=[12],
        fill=False
    )
)


app_ui = ui.page_navbar(
    ui.nav_panel("Forecast", forecast_ui),
    ui.nav_panel("Map", map_ui),
    ui.nav_panel("Data", data_ui),
    ui.nav_panel("About", about_ui),
    ui.nav_panel("Source Code", code_source_ui),
    title=ui.tags.div(
        ui.tags.span("PrevAPP", class_="navbar-brand", id="app-title")
    ),
    id="navbar"
)

app_ui = ui.page_fluid(
    ui.tags.head(
        ui.include_css(Path(__file__).parent / "style.css"),
        ui.tags.style(navbar_style),
        ui.tags.script("""
             document.addEventListener('DOMContentLoaded', function() {
                document.getElementById('app-title').addEventListener('click', function(event) {
                    event.preventDefault();
                    window.location.reload();
                });
            });
        """)
    ),
    app_ui
)

# Serveur de l'application
def server(input, output, session):
    predictions_df = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.visualize_button)
    async def on_visualize_click():
        try:
            date = input.date_input()
            horizon = input.horizon()
            with ui.Progress(min=1, max=15) as p:
                p.set(message="Prediction in progress", detail="in progress... Please wait.")
                for i in range(1, 15):
                    p.set(i, message="Prediction")
                    await asyncio.sleep(0.1)
            predictions_df.set(predict_future_incidence(xgb_model2, df_malaria, 'Incidence/1000', date, horizon))
        except Exception as e:
            ui.notification_show(f"Error: {str(e)}", duration=5, type="error")

    @output
    @render.ui
    def prediction_plot():
        if predictions_df() is None:
            return ui.HTML("<p>Please select the date, the horizon, and click on Visualize.</p>")
        fig = future_predictions12(predictions_df())
        return ui.HTML(fig.to_html(full_html=False))

    @output
    @render.ui
    def date_select():
        if predictions_df() is None:
            return None
        dates = predictions_df()['Date'].dt.strftime('%Y-%m-%d').unique().tolist()
        return ui.input_select("date_choice", "Select a date", choices=[""] + dates)

    @output
    @render.ui
    def map_plot():
        if predictions_df() is None or not input.date_choice():
            return ui.HTML("<p>Please visualize the forecasts first, then select a date to display the map.</p>")
        date_selected = input.date_choice()
        map_object = plot_prediction_map_with_regions(predictions_df(), date_selected)
        map_html = render_folium_map(map_object)
        return ui.HTML(map_html)

    @output
    @render.data_frame
    def data_grid():
        if predictions_df() is None:
            raise Exception("No data to download. Please visualize the forecasts first.")
        return render.DataGrid(predictions_df(), selection_mode="rows")

    @output
    @render.ui
    def rows():
        rows = data_grid.cell_selection()["rows"]
        selected = ", ".join(str(i) for i in sorted(rows)) if rows else "None"
        return f"Rows selected: {selected}"

    @output
    @render.download
    def download_csv():
        if predictions_df() is None:
            raise Exception("No data to download. Please visualize the forecasts first.")

        # Convertir le DataFrame en CSV
        def generate_csv():
            csv_data = predictions_df().to_csv(index=False)
            return csv_data.encode('utf-8')

        return generate_csv()

# Exécution de l'application
www_dir = Path(__file__).parent / "www"
app = shiny.App(app_ui, server,  static_assets=www_dir)
if __name__ == "__main__":
    app.run()

