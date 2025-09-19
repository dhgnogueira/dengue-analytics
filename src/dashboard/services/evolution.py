import plotly.express as px
import streamlit as st


def show_evolution(df_rmvp, municipality_col, municipalities):
    st.header("Evolução temporal dos casos por município e região")
    df_mun_ano = df_rmvp.groupby([municipality_col, "Ano"])["Total"].sum().reset_index()
    municipio_evol = st.selectbox("Selecione o município para evolução temporal", municipalities)
    df_evol = df_mun_ano[df_mun_ano[municipality_col] == municipio_evol]
    fig_evol = px.line(
        df_evol, x="Ano", y="Total", markers=True, title=f"Evolução dos casos em {municipio_evol}"
    )
    st.plotly_chart(fig_evol, use_container_width=True)
