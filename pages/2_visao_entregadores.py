import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import plotly.express as px

# =========================
# Funções utilitárias
# =========================

def carregar_logo(path='imagens/logo.png'):
    """
    Carrega a imagem do logo a partir do caminho especificado.

    Parâmetros:
        path (str): Caminho para o arquivo de imagem.

    Retorna:
        Image: Objeto de imagem PIL.
    """
    return Image.open(path)

def image_to_base64(img):
    """
    Converte uma imagem PIL para uma string codificada em base64.

    Parâmetros:
        img (Image): Objeto de imagem PIL.

    Retorna:
        str: String da imagem em base64.
    """
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def carregar_dados(path='dados/train.csv'):
    """
    Carrega os dados do arquivo CSV para um DataFrame do pandas.

    Parâmetros:
        path (str): Caminho para o arquivo CSV.

    Retorna:
        DataFrame: Dados carregados.
    """
    df = pd.read_csv(path)
    return df

def limpar_dados(df):
    """
    Realiza a limpeza e transformação dos dados conforme regras de negócio.

    Parâmetros:
        df (DataFrame): DataFrame original.

    Retorna:
        DataFrame: DataFrame limpo e transformado.
    """
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    df['Weatherconditions'] = df['Weatherconditions'].apply(
        lambda x: x.replace('conditions ', '') if pd.notna(x) and isinstance(x, str) else x
    )
    df = df.replace(['NaN'], pd.NA)
    df['Delivery_person_Age'] = pd.to_numeric(df['Delivery_person_Age']).astype('Int64')
    df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'])
    df['multiple_deliveries'] = pd.to_numeric(df['multiple_deliveries']).astype('Int64')
    df['Time_taken(min)'] = df['Time_taken(min)'].apply(lambda x: float(str(x).replace('(min) ', '')))
    df = df.dropna(subset=['Delivery_person_Age', 'Road_traffic_density', 'City', 'Festival'])
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y')
    return df

def exibir_logo_sidebar(imagem_logo_base64):
    """
    Exibe o logo na barra lateral do Streamlit usando uma string base64.

    Parâmetros:
        imagem_logo_base64 (str): String base64 da imagem do logo.
    """
    st.sidebar.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{imagem_logo_base64}" style="height: 100px; margin-right: 10px; float: left;" />
        </div>
        """,
        unsafe_allow_html=True
    )

def sidebar_infos(df):
    """
    Exibe e gerencia os filtros na barra lateral do Streamlit, permitindo ao usuário filtrar por data,
    idade do entregador e condição do veículo.

    Parâmetros:
        df (DataFrame): DataFrame de dados limpos.

    Retorna:
        tuple: (date_slider, idade_selecionada, cond_veic_selecionada)
    """
    st.sidebar.markdown('# Curry Company')
    st.sidebar.markdown('## Fastest Delivery in Town')
    st.sidebar.markdown('---')
    st.sidebar.markdown('## Selecione uma data limite')

    min_date = df['Order_Date'].min().date()
    max_date = df['Order_Date'].max().date()
    valor_padrao = max_date if pd.to_datetime('2022-04-13').date() > max_date else pd.to_datetime('2022-04-13').date()

    date_slider = st.sidebar.slider(
        'Até qual valor?',
        value=valor_padrao,
        min_value=min_date,
        max_value=max_date,
        step=pd.Timedelta(days=1)
    )
    st.markdown(date_slider)
    st.sidebar.markdown('---')

    st.sidebar.markdown('## Filtros de Métricas')
    idade_min = int(df['Delivery_person_Age'].min())
    idade_max = int(df['Delivery_person_Age'].max())
    idade_selecionada = st.sidebar.slider(
        'Idade do entregador',
        min_value=idade_min,
        max_value=idade_max,
        value=(idade_min, idade_max),
        step=1
    )

    cond_veic_min = int(df['Vehicle_condition'].min())
    cond_veic_max = int(df['Vehicle_condition'].max())
    cond_veic_selecionada = st.sidebar.slider(
        'Condição do veículo',
        min_value=cond_veic_min,
        max_value=cond_veic_max,
        value=(cond_veic_min, cond_veic_max),
        step=1
    )

    return date_slider, idade_selecionada, cond_veic_selecionada

def aplicar_filtros(df, date_slider, idade_selecionada, cond_veic_selecionada):
    """
    Aplica os filtros selecionados pelo usuário ao DataFrame.

    Parâmetros:
        df (DataFrame): DataFrame de dados limpos.
        date_slider (date): Data limite selecionada.
        idade_selecionada (tuple): Faixa de idade selecionada.
        cond_veic_selecionada (tuple): Faixa de condição do veículo selecionada.

    Retorna:
        DataFrame: DataFrame filtrado conforme os critérios.
    """
    df_filtrado = df[
        (df['Delivery_person_Age'] >= idade_selecionada[0]) &
        (df['Delivery_person_Age'] <= idade_selecionada[1]) &
        (df['Vehicle_condition'] >= cond_veic_selecionada[0]) &
        (df['Vehicle_condition'] <= cond_veic_selecionada[1])
    ]
    df_filtrado = df_filtrado[df_filtrado['Order_Date'].dt.date <= date_slider]
    return df_filtrado

def visao_gerencial(df):
    """
    Exibe a visão gerencial com métricas resumidas e gráficos de avaliação dos entregadores.

    Parâmetros:
        df (DataFrame): DataFrame filtrado.
    """
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        menor_idade = df['Delivery_person_Age'].min()
        st.markdown(
            f'<span style="font-size:16px"><b>A menor idade:</b></span><br>'
            f'<span style="font-size:22px; color:#3a86ff"><b>{menor_idade}</b></span>',
            unsafe_allow_html=True
        )
    with col2:
        maior_idade = df['Delivery_person_Age'].max()
        st.markdown(
            f'<span style="font-size:16px"><b>Maior idade:</b></span><br>'
            f'<span style="font-size:22px; color:#3a86ff"><b>{maior_idade}</b></span>',
            unsafe_allow_html=True
        )
    with col3:
        pior_cond_veiculo = df['Vehicle_condition'].min()
        st.markdown(
            f'<span style="font-size:16px"><b>Pior condição:</b></span><br>'
            f'<span style="font-size:22px; color:#3a86ff"><b>{pior_cond_veiculo}</b></span>',
            unsafe_allow_html=True
        )
    with col4:
        melhor_cond_veiculo = df['Vehicle_condition'].max()
        st.markdown(
            f'<span style="font-size:16px"><b>Melhor condição:</b></span><br>'
            f'<span style="font-size:22px; color:#3a86ff"><b>{melhor_cond_veiculo}</b></span>',
            unsafe_allow_html=True
        )
    st.markdown('---')

    # Gráfico de avaliação média por semana
    df['Semana_do_Ano'] = df['Order_Date'].dt.isocalendar().week
    avaliacao_media_semana = df.groupby('Semana_do_Ano')['Delivery_person_Ratings'].mean().reset_index()
    avaliacao_media_semana = avaliacao_media_semana.rename(columns={
        'Semana_do_Ano': 'Semana do Ano',
        'Delivery_person_Ratings': 'Avaliação Média'
    })
    fig = px.line(
        avaliacao_media_semana,
        x='Semana do Ano',
        y='Avaliação Média',
        markers=True,
        title=''
    )
    fig.update_layout(
        xaxis_title='Semana do Ano',
        yaxis_title='Avaliação Média',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Arial, sans-serif'),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    fig.update_xaxes(
        showgrid=False, 
        zeroline=False, 
        linecolor='white',
        tickfont=dict(color='white', size=10)
    )
    fig.update_yaxes(
        showgrid=False, 
        zeroline=False, 
        linecolor='white',
        tickfont=dict(color='white', size=10)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('---')


    avaliacao_media_trafego_cidade = df.groupby(['Road_traffic_density', 'City'])['Delivery_person_Ratings'].mean().reset_index()
    avaliacao_media_trafego_cidade = avaliacao_media_trafego_cidade.rename(columns={
        'Road_traffic_density': 'Tipo de Tráfego',
        'City': 'Cidade',
        'Delivery_person_Ratings': 'Avaliação Média'
    })
    fig_trafego_cidade = px.bar(
        avaliacao_media_trafego_cidade,
        x='Tipo de Tráfego',
        y='Avaliação Média',
        color='Cidade',
        barmode='group',
        title='',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig_trafego_cidade.update_layout(
        xaxis_title='Tipo de Tráfego',
        yaxis_title='Avaliação Média',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Arial, sans-serif'),
        showlegend=True,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    fig_trafego_cidade.update_xaxes(
        showgrid=False, 
        zeroline=False, 
        linecolor='white',
        tickfont=dict(color='white', size=10)
    )
    fig_trafego_cidade.update_yaxes(
        showgrid=False, 
        zeroline=False, 
        linecolor='white',
        tickfont=dict(color='white', size=10)
    )
    st.plotly_chart(fig_trafego_cidade, use_container_width=True)
    st.markdown('---')

def visao_meritocratica(df):
    """
    Exibe a visão meritocrática, mostrando rankings dos entregadores mais rápidos e mais lentos.

    Parâmetros:
        df (DataFrame): DataFrame filtrado.
    """
    st.markdown("### Ranking dos Entregadores")

    st.markdown("#### Entregadores Mais Rápidos")
    entregadores_rapidos = (
        df.groupby('Delivery_person_ID')['Time_taken(min)']
        .mean()
        .reset_index()
        .sort_values('Time_taken(min)', ascending=True)
        .head(10)
    )
    entregadores_rapidos = entregadores_rapidos.rename(columns={
        'Delivery_person_ID': 'ID do Entregador',
        'Time_taken(min)': 'Tempo Médio de Entrega (min)'
    })
    entregadores_rapidos = entregadores_rapidos[['Tempo Médio de Entrega (min)', 'ID do Entregador']]
    st.dataframe(entregadores_rapidos, use_container_width=True)
    st.markdown('---')

    st.markdown("#### Entregadores Mais Lentos")
    entregadores_lentos = (
        df.groupby('Delivery_person_ID')['Time_taken(min)']
        .mean()
        .reset_index()
        .sort_values('Time_taken(min)', ascending=False)
        .head(10)
    )
    entregadores_lentos = entregadores_lentos.rename(columns={
        'Delivery_person_ID': 'ID do Entregador',
        'Time_taken(min)': 'Tempo Médio de Entrega (min)'
    })
    entregadores_lentos = entregadores_lentos[['Tempo Médio de Entrega (min)', 'ID do Entregador']]
    st.dataframe(entregadores_lentos, use_container_width=True)
    st.markdown('---')

# =========================
# Execução principal
# =========================

def main():
    """
    Função principal que executa o fluxo da aplicação Streamlit:
    - Carrega e limpa os dados
    - Exibe o logo
    - Mostra os filtros na barra lateral
    - Aplica os filtros
    - Exibe as abas de visão gerencial e meritocrática
    """
    imagem_logo = carregar_logo()
    imagem_logo_base64 = image_to_base64(imagem_logo)
    df = carregar_dados()
    df = limpar_dados(df)

    st.header('Marketplace - Visão Entregadores')
    exibir_logo_sidebar(imagem_logo_base64)
    date_slider, idade_selecionada, cond_veic_selecionada = sidebar_infos(df)
    df_filtrado = aplicar_filtros(df, date_slider, idade_selecionada, cond_veic_selecionada)

    tab1, tab2 = st.tabs(['Visão Gerencial', 'Visão Meritocrática'])

    with tab1:
        visao_gerencial(df_filtrado)
    with tab2:
        visao_meritocratica(df_filtrado)

if __name__ == "__main__":
    main()
