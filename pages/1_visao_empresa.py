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
    Carrega a imagem do logo da empresa a partir do caminho especificado.

    Args:
        path (str): Caminho para o arquivo de imagem do logo.

    Returns:
        PIL.Image: Objeto de imagem carregado.
    """
    return Image.open(path)

def image_to_base64(img):
    """
    Converte uma imagem PIL para uma string codificada em base64.

    Args:
        img (PIL.Image): Imagem a ser convertida.

    Returns:
        str: String base64 da imagem.
    """
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def carregar_dados(path='dados/train.csv'):
    """
    Carrega os dados brutos do arquivo CSV em um DataFrame.

    Args:
        path (str): Caminho para o arquivo CSV.

    Returns:
        pd.DataFrame: DataFrame com os dados carregados.
    """
    df = pd.read_csv(path)
    return df

def limpar_dados(df):
    """
    Realiza a limpeza e transformação dos dados conforme regras de negócio.

    - Remove espaços em branco de colunas de texto.
    - Ajusta valores de condições climáticas.
    - Converte colunas para tipos numéricos apropriados.
    - Remove linhas com valores ausentes em colunas essenciais.
    - Converte datas para o formato datetime.

    Args:
        df (pd.DataFrame): DataFrame bruto.

    Returns:
        pd.DataFrame: DataFrame limpo e transformado.
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
    Exibe o logo da empresa na barra lateral do Streamlit.

    Args:
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
    Exibe e gerencia os filtros interativos na barra lateral do Streamlit.

    - Filtro de data limite.
    - Filtro de faixa etária dos entregadores.
    - Filtro de festivais.

    Args:
        df (pd.DataFrame): DataFrame limpo.

    Returns:
        tuple: (date_slider, idade_selecionada, festival_options)
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
    st.sidebar.markdown('---')

    
    idade_min = int(df['Delivery_person_Age'].min())
    idade_max = int(df['Delivery_person_Age'].max())
    idade_selecionada = st.sidebar.slider(
        'Idade do entregador',
        min_value=idade_min,
        max_value=idade_max,
        value=(idade_min, idade_max),
        step=1
    )
    st.sidebar.markdown('---')

    st.sidebar.markdown('## Selecione Festival')
    festival_options = st.sidebar.multiselect(
        'Festival?',
        options=sorted(df['Festival'].dropna().unique()),
        default=sorted(df['Festival'].dropna().unique())
    )
    st.sidebar.markdown('---')
    return date_slider, idade_selecionada, festival_options

def aplicar_filtros(df, date_slider, idade_selecionada, festival_options):
    """
    Aplica os filtros selecionados pelo usuário ao DataFrame.

    - Filtra por data limite.
    - Filtra por faixa etária dos entregadores.
    - Filtra por festivais selecionados.

    Args:
        df (pd.DataFrame): DataFrame limpo.
        date_slider (datetime.date): Data limite selecionada.
        idade_selecionada (tuple): Faixa etária selecionada (min, max).
        festival_options (list): Lista de festivais selecionados.

    Returns:
        pd.DataFrame: DataFrame filtrado.
    """
    # Filtra pela data selecionada
    df_filtros = df[df['Order_Date'].dt.date <= date_slider]

    # Filtro de idade dos entregadores
    idade_min, idade_max = idade_selecionada
    df_filtros = df_filtros[
        (df_filtros['Delivery_person_Age'] >= idade_min) &
        (df_filtros['Delivery_person_Age'] <= idade_max)
    ]


    if festival_options:
        df_filtros = df_filtros[df_filtros['Festival'].isin(festival_options)]

    return df_filtros

# =========================
# Funções de visualização
# =========================

def grafico_pedidos_por_dia(df_filtros):
    """
    Exibe um gráfico de barras com a quantidade de pedidos por dia.

    Args:
        df_filtros (pd.DataFrame): DataFrame filtrado.
    """
    pedidos_por_dia = df_filtros.groupby('Order_Date')['ID'].nunique().reset_index(name='Quantidade de Pedidos')
    pedidos_por_dia = pedidos_por_dia.sort_values('Order_Date')
    fig = px.bar(
        pedidos_por_dia,
        x='Order_Date',
        y='Quantidade de Pedidos',
        title="Pedidos por Dia",
        color_discrete_sequence=['#3a86ff']
    )
    fig.update_layout(
        xaxis_title='Data do Pedido',
        yaxis_title='Quantidade de Pedidos',
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

def grafico_pizza_trafego(df_filtros):
    """
    Exibe um gráfico de pizza mostrando a distribuição dos pedidos por tipo de tráfego.

    Args:
        df_filtros (pd.DataFrame): DataFrame filtrado.
    """
    pedidos_por_trafego = df_filtros['Road_traffic_density'].value_counts().reset_index()
    pedidos_por_trafego.columns = ['Road_traffic_density', 'Quantidade de Pedidos']
    pedidos_por_trafego = pedidos_por_trafego[pedidos_por_trafego['Road_traffic_density'].notna()]
    labels = pedidos_por_trafego['Road_traffic_density']
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='none')
    wedges, texts, autotexts = ax.pie(
        pedidos_por_trafego['Quantidade de Pedidos'],
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'color': 'white'}
    )
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    for text in texts + autotexts:
        text.set_color('white')
    ax.axis('equal')
    ax.set_title("Distribuição dos Pedidos por Tipo de Tráfego", color='white', fontsize=14)
    st.pyplot(fig)

def grafico_pedidos_por_cidade(df_filtros):
    """
    Exibe um gráfico de barras com a quantidade de pedidos por cidade.

    Args:
        df_filtros (pd.DataFrame): DataFrame filtrado.
    """
    pedidos_por_cidade = df_filtros.groupby('City')['ID'].nunique().reset_index(name='Quantidade de Pedidos')
    pedidos_por_cidade = pedidos_por_cidade.sort_values('Quantidade de Pedidos', ascending=False)
    fig = px.bar(
        pedidos_por_cidade,
        x='City',
        y='Quantidade de Pedidos',
        title="Pedidos por Cidade",
        color_discrete_sequence=['#3a86ff']
    )
    fig.update_layout(
        xaxis_title='Cidade',
        yaxis_title='Quantidade de Pedidos',
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

def grafico_bolha_cidade_trafego(df_filtros):
    """
    Exibe um gráfico de bolhas relacionando cidade, tipo de tráfego e quantidade de pedidos.

    Args:
        df_filtros (pd.DataFrame): DataFrame filtrado.
    """
    df_bolha = df_filtros.copy()
    if df_bolha['Road_traffic_density'].dtype in ['int64', 'float64']:
        mapa_trafego = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Jam'}
        df_bolha['Road_traffic_density'] = df_bolha['Road_traffic_density'].map(mapa_trafego)
    df_bolha = df_bolha.dropna(subset=['City', 'Road_traffic_density'])
    agrupado = df_bolha.groupby(['City', 'Road_traffic_density'])['ID'].nunique().reset_index(name='Quantidade')
    agrupado = agrupado.sort_values(['City', 'Road_traffic_density'])
    cores_azuis_diferenciaveis = ['#3a86ff', '#4361ee', '#023e8a', '#48cae4']
    fig = px.scatter(
        agrupado,
        x='City',
        y='Road_traffic_density',
        size='Quantidade',
        color='Road_traffic_density',
        title="Pedidos por Cidade e Tipo de Tráfego",
        size_max=60,
        color_discrete_sequence=cores_azuis_diferenciaveis
    )
    fig.update_traces(marker=dict(line=dict(width=1, color='white')))
    fig.update_layout(
        xaxis_title='Cidade',
        yaxis_title='Tipo de Tráfego',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Arial, sans-serif'),
        showlegend=True,
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

def grafico_linha_pedidos_semana(df_filtros):
    """
    Exibe um gráfico de linha com a quantidade de pedidos por semana do ano.

    Args:
        df_filtros (pd.DataFrame): DataFrame filtrado.

    Returns:
        pd.DataFrame: DataFrame com pedidos por semana (para uso em outros gráficos).
    """
    isocal = df_filtros['Order_Date'].dt.isocalendar()
    pedidos_por_semana = (
        df_filtros.groupby([isocal['year'], isocal['week']])['ID']
        .nunique()
        .reset_index()
        .rename(columns={'ID': 'id', 'year': 'Ano', 'week': 'Numero_Semana'})
    )
    pedidos_por_semana['Ano_Semana'] = pedidos_por_semana['Ano'].astype(str) + '-S' + pedidos_por_semana['Numero_Semana'].astype(str).str.zfill(2)
    fig = px.line(
        pedidos_por_semana,
        x='Ano_Semana',
        y='id',
        labels={'Ano_Semana': 'Ano-Semana', 'id': 'Quantidade de Pedidos'},
        title="Pedidos por Semana",
        markers=True,
        color_discrete_sequence=['#00bcd4']
    )
    fig.update_layout(
        xaxis_title='Ano-Semana',
        yaxis_title='Quantidade de Pedidos',
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
    return pedidos_por_semana

def grafico_barra_pedidos_semana(pedidos_por_semana):
    """
    Exibe um gráfico de barras com a quantidade de pedidos por semana do ano.

    Args:
        pedidos_por_semana (pd.DataFrame): DataFrame com pedidos por semana.
    """
    fig = px.bar(
        pedidos_por_semana,
        x='Ano_Semana',
        y='id',
        labels={'Ano_Semana': 'Ano-Semana', 'id': 'Quantidade de Pedidos'},
        title="Pedidos por Semana (Barras)",
        color_discrete_sequence=['#00bcd4']
    )
    fig.update_layout(
        xaxis_title='Ano-Semana',
        yaxis_title='Quantidade de Pedidos',
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

def grafico_media_pedidos_entregador_semana(df_filtros):
    """
    Exibe um gráfico de linha mostrando a média de pedidos por entregador em cada semana.

    Args:
        df_filtros (pd.DataFrame): DataFrame filtrado.
    """
    order_date = pd.to_datetime(df_filtros['Order_Date'])
    semana = order_date.dt.isocalendar().week
    pedidos_por_semana = df_filtros.groupby(semana)['ID'].count()
    entregadores_por_semana = df_filtros.groupby(semana)['Delivery_person_ID'].nunique()
    media = (pedidos_por_semana / entregadores_por_semana).reset_index(name='Média_Pedidos')
    media.rename(columns={'week': 'Semana'}, inplace=True)
    fig = px.line(
        media,
        x='Semana',
        y='Média_Pedidos',
        markers=True,
        title="Média de Pedidos por Entregador por Semana",
        color_discrete_sequence=['#00bcd4']
    )
    fig.update_layout(
        xaxis_title='Semana do Ano',
        yaxis_title='Média de Pedidos por Entregador',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Arial, sans-serif'),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        title=''
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

# =========================
# Main
# =========================

def main():
    """
    Função principal que executa o fluxo da aplicação Streamlit.

    - Carrega e limpa os dados.
    - Exibe o logo e os filtros na barra lateral.
    - Aplica os filtros selecionados.
    - Exibe as abas de visualização com os respectivos gráficos.
    """
    imagem_logo = carregar_logo()
    imagem_logo_base64 = image_to_base64(imagem_logo)
    df_raw = carregar_dados()
    df = limpar_dados(df_raw)
    st.header('Marketplace - Visão Empresa')
    exibir_logo_sidebar(imagem_logo_base64)
    # Corrigido: sidebar_infos e aplicar_filtros só aceitam 3 argumentos de retorno
    date_slider, idade_selecionada, festival_options = sidebar_infos(df)
    df_filtros = aplicar_filtros(df, date_slider, idade_selecionada, festival_options)
    tab1, tab2 = st.tabs(['Visão Gerencial', 'Visão Tática'])

    with tab1:
        with st.container():
            col1 = st.columns(1)
            with col1[0]:
                grafico_pedidos_por_dia(df_filtros)
            st.markdown('---')
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                grafico_pizza_trafego(df_filtros)
            with col2_2:
                grafico_pedidos_por_cidade(df_filtros)
            st.markdown('---')
            grafico_bolha_cidade_trafego(df_filtros)
            st.markdown('---')

    with tab2:
        st.title('Visão Tática')
        pedidos_por_semana = grafico_linha_pedidos_semana(df_filtros)
        st.markdown('---')
        grafico_barra_pedidos_semana(pedidos_por_semana)
        st.markdown('---')
        grafico_media_pedidos_entregador_semana(df_filtros)
        st.markdown('---')

if __name__ == "__main__":
    main()