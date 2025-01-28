import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import folium
from streamlit_folium import st_folium
import networkx as nx

def carregar_e_processar_dados(file):
    """
    Carrega e processa o CSV com dados de poluição e trânsito
    Espera colunas: latitude, longitude, poluicao, transito
    """
    df = pd.read_csv(file)
    
    # Normaliza os dados de poluição e trânsito para escala 0-1
    scaler = MinMaxScaler()
    df[['poluicao_norm', 'transito_norm']] = scaler.fit_transform(df[['poluicao', 'transito']])
    
    # Calcula score composto (quanto menor, melhor)
    df['score'] = (df['poluicao_norm'] + df['transito_norm']) / 2
    
    return df

def encontrar_melhor_circuito(df, num_pontos=10):
    """
    Encontra um circuito otimizado usando os pontos com menores scores
    """
    # Seleciona os melhores pontos
    melhores_pontos = df.nlargest(num_pontos, 'score')
    
    # Cria um grafo
    G = nx.Graph()
    
    # Adiciona nós
    for idx, row in melhores_pontos.iterrows():
        G.add_node(idx, pos=(row['latitude'], row['longitude']))
    
    # Adiciona arestas com pesos baseados na distância
    for idx1, row1 in melhores_pontos.iterrows():
        for idx2, row2 in melhores_pontos.iterrows():
            if idx1 != idx2:
                dist = np.sqrt((row1['latitude'] - row2['latitude'])**2 + 
                             (row1['longitude'] - row2['longitude'])**2)
                G.add_edge(idx1, idx2, weight=dist)
    
    # Encontra o ciclo hamiltoniano aproximado
    circuito = nx.approximation.traveling_salesman_problem(G, cycle=True)
    
    return melhores_pontos.loc[circuito]

def criar_mapa(df_circuito):
    """
    Cria um mapa interativo com o circuito
    """
    # Cria o mapa centralizado na média dos pontos
    centro = [df_circuito['latitude'].mean(), df_circuito['longitude'].mean()]
    m = folium.Map(location=centro, zoom_start=13)
    
    # Adiciona marcadores para cada ponto
    for idx, row in df_circuito.iterrows():
        folium.Marker(
            [row['latitude'], row['longitude']],
            popup=f"Poluição: {row['poluicao']:.2f}<br>Trânsito: {row['transito']:.2f}",
            tooltip=f"Ponto {idx}"
        ).add_to(m)
    
    # Desenha o circuito
    pontos = df_circuito[['latitude', 'longitude']].values.tolist()
    pontos.append(pontos[0])  # Fecha o circuito
    folium.PolyLine(
        pontos,
        weight=2,
        color='red',
        opacity=0.8
    ).add_to(m)
    
    return m

def main():
    st.title('Otimizador de Circuito - Áreas Limpas')
    
    # Upload do arquivo
    uploaded_file = st.file_uploader("Escolha seu arquivo CSV", type="csv")
    
    if uploaded_file is not None:
        # Carrega e processa os dados
        df = carregar_e_processar_dados(uploaded_file)
        
        # Slider para número de pontos
        num_pontos = st.slider('Número de pontos no circuito', 5, 20, 10)
        
        # Encontra o melhor circuito
        df_circuito = encontrar_melhor_circuito(df, num_pontos)
        
        # Cria e exibe o mapa
        mapa = criar_mapa(df_circuito)
        st_folium(mapa, width=800)
        
        # Exibe estatísticas
        st.subheader('Estatísticas do Circuito')
        col1, col2 = st.columns(2)
        with col1:
            st.metric('Média de Poluição', f"{df_circuito['poluicao'].mean():.2f}")
        with col2:
            st.metric('Média de Trânsito', f"{df_circuito['transito'].mean():.2f}")
        
        # Exibe os dados do circuito
        st.subheader('Pontos do Circuito')
        st.dataframe(df_circuito[['latitude', 'longitude', 'poluicao', 'transito']])

if __name__ == '__main__':
    main()
