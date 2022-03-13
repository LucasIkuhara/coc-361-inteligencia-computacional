import plotly.express as px
import plotly.figure_factory as ff


# Função de plot de matriz de confusão
def plot_confusion(confusion_matrix, title: str = 'Confusion Matrix'):
    '''
    # plot_confusion
    Plotar matriz de confusão.

    Args:
    confusion_matrix - matriz de confusão
    title? - título do gráfico
    Name - nome do modelo
    '''

    # Inverter campos
    confusion_matrix = confusion_matrix[::-1]

    # Montar anotações de célula
    # https://stackoverflow.com/questions/60860121/plotly-how-to-make-an-annotated-confusion-matrix-using-a-heatmap
    annotations = [[str(y) for y in x] for x in confusion_matrix]

    # Nomear campos
    x_labels = ['não-venda', 'venda']
    y_labels = ['venda', 'não-venda']

    # confusion_matrix
    fig = ff.create_annotated_heatmap(
                                      confusion_matrix,
                                      x=x_labels, y=y_labels,
                                      annotation_text=annotations,
                                      colorscale='Blues'
                                      )

    # Títulos
    fig.update_layout(title_text=title)
    fig.update_xaxes(title_text='Valores Referência')
    fig.update_yaxes(title_text='Predições')

    # Barra de cor
    fig['data'][0]['showscale'] = True

    fig.show(config={
        'displaylogo': False,
        'scrollZoom': True
    })
