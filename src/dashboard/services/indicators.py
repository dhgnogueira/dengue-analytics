import streamlit as st


def show_indicators(df_rmvp, municipality_col):
    df_validos = df_rmvp[
        (df_rmvp[municipality_col].notnull())
        & (df_rmvp[municipality_col] != "0")
        & (df_rmvp[municipality_col] != "")
    ]

    def format_value(value):
        if value >= 1_000_000:
            return f"{value / 1_000_000:.2f} milhões"
        elif value >= 1_000:
            return f"{value / 1_000:.0f} mil"
        else:
            return f"{value}"

    total_casos = int(df_validos["Total"].sum())
    media_anual = int(df_validos.groupby("Ano")["Total"].sum().mean())
    pico_ano = df_validos.groupby("Ano")["Total"].sum().idxmax()
    pico_valor = int(df_validos.groupby("Ano")["Total"].sum().max())
    muni_pico = df_validos.groupby(municipality_col)["Total"].sum().idxmax()
    muni_pico_valor = int(df_validos.groupby(municipality_col)["Total"].sum().max())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de casos", format_value(total_casos))
    col2.metric("Média anual de casos", format_value(media_anual))
    col3.metric("Ano com mais casos", f"{pico_ano} ({format_value(pico_valor)})")
    col4.metric(
        "Município com mais casos (total acumulado)",
        f"{muni_pico} ({format_value(muni_pico_valor)})",
    )
