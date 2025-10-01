import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go

# =========================
# Funções utilitárias
# =========================

def carregar_logo(path='imagens/logo.png'):
    """
    Carrega a imagem do logo da empresa a partir do caminho especificado.

    Parâmetros:
        path (str): Caminho para o arquivo de imagem do logo.

    Retorna:
        PIL.Image: Objeto de imagem carregado.
    """
    return Image.open(path)

def image_to_base64(img):
    """
    Converte uma imagem PIL para uma string codificada em base64.

    Parâmetros:
        img (PIL.Image): Imagem a ser convertida.

    Retorna:
        str: String base64 da imagem.
    """
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def carregar_dados(path='dados/train.csv'):
    """
    Carrega os dados brutos do arquivo CSV em um DataFrame.

    Parâmetros:
        path (str): Caminho para o arquivo CSV.

    Retorna:
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

    Parâmetros:
        df (pd.DataFrame): DataFrame bruto.

    Retorna:
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

def sidebar_filtros(df):
    """
    Exibe e gerencia os filtros na barra lateral do Streamlit, permitindo ao usuário filtrar por data,
    idade do entregador e condição do veículo.

    Parâmetros:
        df (pd.DataFrame): DataFrame de dados limpos.

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
        df (pd.DataFrame): DataFrame de dados limpos.
        date_slider (datetime.date): Data limite selecionada.
        idade_selecionada (tuple): Faixa de idade selecionada.
        cond_veic_selecionada (tuple): Faixa de condição do veículo selecionada.

    Retorna:
        pd.DataFrame: DataFrame filtrado conforme os critérios.
    """
    df_filtrado = df[
        (df['Delivery_person_Age'] >= idade_selecionada[0]) &
        (df['Delivery_person_Age'] <= idade_selecionada[1]) &
        (df['Vehicle_condition'] >= cond_veic_selecionada[0]) &
        (df['Vehicle_condition'] <= cond_veic_selecionada[1])
    ]
    df_filtrado = df_filtrado[df_filtrado['Order_Date'].dt.date <= date_slider]
    return df_filtrado

def haversine(lat1, lon1, lat2, lon2):
    """
    Calcula a distância em quilômetros entre dois pontos geográficos usando a fórmula de Haversine.

    Parâmetros:
        lat1, lon1: Latitude e longitude do ponto 1.
        lat2, lon2: Latitude e longitude do ponto 2.

    Retorna:
        float ou np.array: Distância em quilômetros.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

def mostrar_metricas(df):
    """
    Exibe métricas principais no dashboard, como número de entregadores únicos,
    distância média, tempo médio de entrega com e sem festival.

    Parâmetros:
        df (pd.DataFrame): DataFrame filtrado.
    """
    num_entregadores = df['Delivery_person_ID'].nunique()
    df['distancia'] = haversine(
        df['Restaurant_latitude'], df['Restaurant_longitude'],
        df['Delivery_location_latitude'], df['Delivery_location_longitude']
    )
    distancia_media = df['distancia'].mean()
    tempo_medio_com_festival = df[df['Festival'] == 'Yes']['Time_taken(min)'].mean()
    tempo_medio_sem_festival = df[df['Festival'] == 'No']['Time_taken(min)'].mean()

    col1, espaco1, col2, espaco2, col3, espaco3, col4 = st.columns([2, 0.5, 2, 0.5, 2, 0.5, 2])
    with col1:
        st.markdown(f"<b>Entregadores únicos:</b><br><span style='font-size:22px; color:#3a86ff'><b>{num_entregadores}</b></span>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<b>Distância média (km):</b><br><span style='font-size:22px; color:#3a86ff'><b>{distancia_media:.2f}</b></span>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<b>Tempo médio c/ festival:</b><br><span style='font-size:22px; color:#3a86ff'><b>{tempo_medio_com_festival:.2f}</b></span>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<b>Tempo médio s/ festival:</b><br><span style='font-size:22px; color:#3a86ff'><b>{tempo_medio_sem_festival:.2f}</b></span>", unsafe_allow_html=True)

def grafico_pizza_distancia(df):
    """
    Exibe um gráfico de pizza mostrando a distância média das entregas por cidade.

    Parâmetros:
        df (pd.DataFrame): DataFrame filtrado, já com coluna 'distancia' calculada.
    """
    st.markdown("---")
    st.markdown("#### Gráfico de pizza: Distância média das entregas por cidade")
    col7 = st.columns(1)[0]
    with col7:
        distancia_media_por_cidade = df.groupby('City')['distancia'].mean().reset_index()
        distancia_media_por_cidade = distancia_media_por_cidade.sort_values('distancia')
        azuis = ['#1f77b4', '#2a9fd6', '#005c99', '#3399ff', '#003366']
        fig = px.pie(
            distancia_media_por_cidade,
            names='City',
            values='distancia',
            color_discrete_sequence=azuis
        )
        fig.update_traces(pull=[0.15 if i == 0 else 0 for i in range(len(distancia_media_por_cidade))])
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Arial, sans-serif'),
            title=''
        )
        st.plotly_chart(fig, use_container_width=True)

def graficos_tempo_entrega(df):
    """
    Exibe dois gráficos:
    1. Gráfico de barras do tempo médio de entrega por cidade, com desvio padrão.
    2. Gráfico sunburst do desvio padrão do tempo de entrega por cidade e densidade de trânsito.

    Parâmetros:
        df (pd.DataFrame): DataFrame filtrado.
    """
    st.markdown("---")
    st.markdown("#### Gráfico de barras: Tempo médio de entrega por cidade (com desvio padrão)")
    col8, col9 = st.columns(2)

    # Gráfico 1: Distribuição do Tempo de Entrega por Cidade (com Desvio Padrão)
    tempo_cidade = df.groupby('City')['Time_taken(min)'].agg(['mean', 'std']).reset_index()
    tempo_cidade.columns = ['Cidade', 'Tempo Médio de Entrega', 'Desvio Padrão']

    fig1 = go.Figure(
        data=[
            go.Bar(
                x=tempo_cidade['Cidade'],
                y=tempo_cidade['Tempo Médio de Entrega'],
                error_y=dict(
                    type='data',
                    array=tempo_cidade['Desvio Padrão'],
                    visible=True,
                    color='orange',
                    thickness=2,
                    width=8
                ),
                marker_color='royalblue',
                name='Tempo Médio de Entrega'
            )
        ]
    )
    fig1.update_layout(
        title='',
        xaxis_title='Cidade',
        yaxis_title='Tempo Médio de Entrega (min)',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Arial, sans-serif'),
        title_font=dict(size=18, color='white')
    )


    st.markdown("#### Gráfico sunburst: Desvio padrão do tempo de entrega por cidade e trânsito")
    desvio_cidade_trafego = (
        df.groupby(['City', 'Road_traffic_density'])['Time_taken(min)']
        .std()
        .reset_index()
    )
    desvio_cidade_trafego.columns = ['Cidade', 'Trânsito', 'Desvio Padrão']
    desvio_cidade_trafego['Desvio Padrão'] = desvio_cidade_trafego['Desvio Padrão'].fillna(0)

    fig2 = px.sunburst(
        desvio_cidade_trafego,
        path=['Cidade', 'Trânsito'],
        values='Desvio Padrão',
        color='Desvio Padrão',
        color_continuous_scale='viridis',
        title=' '
    )
    fig2.update_layout(
        template='plotly_dark',
        font=dict(color='white', family='Arial, sans-serif'),
        title_font=dict(size=18, color='white'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    with col8:
        st.plotly_chart(fig1, use_container_width=True)
    with col9:
        st.plotly_chart(fig2, use_container_width=True)

def tabela_tempo_cidade_tipo(df):
    """
    Exibe uma tabela com o tempo médio de entrega e desvio padrão agrupados por cidade e tipo de pedido.

    Parâmetros:
        df (pd.DataFrame): DataFrame filtrado.
    """
    st.markdown("---")
    st.markdown("#### Tabela: Tempo médio de entrega e desvio padrão por cidade e tipo de pedido")
    col10 = st.columns(1)[0]
    with col10:
        tempo_cidade_tipo = df.groupby(['City', 'Type_of_order'])['Time_taken(min)'].agg(['mean', 'std']).reset_index()
        tempo_cidade_tipo.columns = ['Cidade', 'Tipo de Pedido', 'Tempo Médio de Entrega', 'Desvio Padrão']
        st.dataframe(tempo_cidade_tipo, use_container_width=True)

# =========================
# Execução principal
# =========================

def main():
    """
    Função principal que executa o fluxo da aplicação Streamlit para a visão de restaurantes.
    Carrega dados, aplica filtros, exibe métricas e gráficos.
    """
    imagem_logo = carregar_logo('imagens/logo.png')
    imagem_logo_base64 = image_to_base64(imagem_logo)
    df = carregar_dados('dados/train.csv')
    df = limpar_dados(df)

    st.header('Marketplace - Visão Restaurantes')
    exibir_logo_sidebar(imagem_logo_base64)
    date_slider, idade_selecionada, cond_veic_selecionada = sidebar_filtros(df)
    df = aplicar_filtros(df, date_slider, idade_selecionada, cond_veic_selecionada)
    mostrar_metricas(df)
    grafico_pizza_distancia(df)
    graficos_tempo_entrega(df)
    tabela_tempo_cidade_tipo(df)

if __name__ == "__main__":
    main()
