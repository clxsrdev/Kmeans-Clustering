import os
from shiny import App, reactive, render, ui
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# Crear un directorio para guardar las imágenes si no existe
output_image_dir = "/Users/chemapadillafdez/Documents/GitHub/Kmeans-Clustering/output_images"
os.makedirs(output_image_dir, exist_ok=True)

# Cargar datos
data = pd.read_csv("/Users/chemapadillafdez/Documents/GitHub/Kmeans-Clustering/Wall.csv")
css_path = "/Users/chemapadillafdez/Documents/GitHub/Kmeans-Clustering/styles.css"

app_ui = ui.page_fluid(
    ui.include_css(css_path),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_slider("age_range", "Edad:", min=data['Age'].min(), max=data['Age'].max(), value=[data['Age'].min(), data['Age'].max()]),
            ui.input_checkbox_group("payment_methods", "Métodos de Pago:",
                                    choices=["Cash", "Tcredit", "Tdebit"],
                                    selected=["Cash", "Tcredit", "Tdebit"]),
            ui.input_slider("n_clusters", "Cluster count", min=2, max=10, value=4)
        ),
        ui.panel_main(
            ui.row(
                ui.column(4, ui.div(ui.output_text_verbatim("num_online"), class_="stat-box")),
                ui.column(4, ui.div(ui.output_text_verbatim("avg_age_effective"), class_="stat-box")),
                ui.column(4, ui.div(ui.output_text_verbatim("avg_age_high_income"), class_="stat-box"))
            ),
            ui.div(
                ui.h2("Método del Codo"),
                ui.div(ui.output_image("elbow_plot"), class_="plot-container"),
                class_="centered"
            ),
            ui.div(
                ui.h2("Segmentación: Medio de Pago"),
                ui.div(ui.output_image("cluster_plot"), class_="plot-container"),
                class_="centered-seg"
            ),
            ui.div(ui.h2("Datos Filtrados"), class_="centered-seg"),
            ui.div(ui.output_data_frame("data_table"), class_="table-container")
        )
    )
)

def server(input, output, session):
    @reactive.Calc
    def filtered_data():
        df = data.copy()
        df = df[df['Age'].between(*input.age_range())]
        df = df[df['Payment_Methods'].isin(input.payment_methods())]
        return df

    @output
    @render.text
    def num_online():
        df = filtered_data()
        if 'OnlinePurchase' in df.columns:
            return f"Número de consumidores online: {df['OnlinePurchase'].sum()}"
        else:
            return "Error: 'OnlinePurchase' no encontrado"

    @output
    @render.text
    def avg_age_effective():
        df = filtered_data()
        if not df[df['Payment_Methods'] == 'Cash'].empty:
            avg_age = df[df['Payment_Methods'] == 'Cash']['Age'].mean()
            return f"Edad promedio para 'Efectivo': {avg_age:.2f}"
        else:
            return "Edad promedio para 'Efectivo': nan"

    @output
    @render.text
    def avg_age_high_income():
        df = filtered_data()
        df_high_income = df[df['Annual_Income'] > 20]
        if not df_high_income.empty:
            avg_age = df_high_income['Age'].mean()
            return f"Edad promedio para ingresos > $20,000: {avg_age:.2f}"
        else:
            return "Edad promedio para ingresos > $20,000: nan"

    @output
    @render.image
    def elbow_plot():
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(data[['Age', 'Annual_Income']])
            wcss.append(kmeans.inertia_)
        fig = go.Figure(data=go.Scatter(x=list(range(1, 11)), y=wcss, mode='lines+markers'))
        fig.update_layout(title='Método del Codo', xaxis_title='Número de Clusters', yaxis_title='WCSS')

        # Guardar la imagen en el directorio conocido
        elbow_plot_path = os.path.join(output_image_dir, "elbow_plot.png")
        fig.write_image(elbow_plot_path)

        return {"src": elbow_plot_path, "alt": "Método del Codo"}

    @output
    @render.image
    def cluster_plot():
        df = filtered_data()
        if df[['Age', 'Annual_Income']].empty:
            return None
        kmeans = KMeans(n_clusters=input.n_clusters(), random_state=0)
        df['Cluster'] = kmeans.fit_predict(df[['Age', 'Annual_Income']])
        fig = px.scatter(df, x='Annual_Income', y='Age', color='Cluster', symbol='Payment_Methods', title='Segmentación: Medio de Pago')
        
        # Ajustar la posición de la leyenda para evitar solapamiento
        fig.update_layout(legend=dict(
            yanchor="top",
            y=1.1,
            xanchor="left",
            x=1.2
        ))

        # Guardar la imagen en el directorio conocido
        cluster_plot_path = os.path.join(output_image_dir, "cluster_plot.png")
        fig.write_image(cluster_plot_path)

        return {"src": cluster_plot_path, "alt": "Segmentación: Medio de Pago"}

    @output
    @render.data_frame
    def data_table():
        # Convertir los valores booleanos a texto para asegurar que se muestran
        df = filtered_data().copy()
        df['OnlinePurchase'] = df['OnlinePurchase'].astype(str)
        return df

app = App(app_ui, server)

if __name__ == "__main__":
    app.run(port=5008)
