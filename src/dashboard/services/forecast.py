import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression


def show_forecast(df_rmvp):
    st.subheader("Previsão de casos de dengue (Regressão Linear)")
    df_ano = df_rmvp.groupby("Ano")["Total"].sum().reset_index()
    df_ano["Ano_num"] = pd.to_numeric(df_ano["Ano"])
    X = df_ano[["Ano_num"]]
    y = df_ano["Total"]
    model = LinearRegression()
    model.fit(X, y)
    anos_futuros = np.arange(df_ano["Ano_num"].min(), df_ano["Ano_num"].max() + 3).reshape(-1, 1)
    anos_futuros_df = pd.DataFrame(anos_futuros, columns=["Ano_num"])
    casos_previstos = model.predict(anos_futuros_df)
    fig3 = px.scatter(
        df_ano,
        x="Ano_num",
        y="Total",
        color_discrete_sequence=["blue"],
        labels={"Ano_num": "Ano", "Total": "Casos"},
        title="Casos Reais e Previstos",
    )
    fig3.add_traces(
        px.line(
            x=anos_futuros_df["Ano_num"],
            y=casos_previstos,
            labels={"x": "Ano", "y": "Casos"},
            color_discrete_sequence=["red"],
        ).data
    )
    fig3.update_layout(showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
