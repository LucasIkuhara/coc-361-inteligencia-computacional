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


# Calcular precisão, recall e F1
def additional_metrics(confusion_matrix):
    '''
    # additional_metrics
    Calcula a precisão, recall e score F1 a partir de uma
    matriz de confusão.

    Args:
        confusion_matriz: matriz de confusão do modelo

    Returns:
        tuple: (precision, recall, f1)
    '''
    tp, fn = confusion_matrix[0]
    fp, fn = confusion_matrix[1]

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*precision*recall / (precision + recall)

    print(f'''
    Precisão: {precision}
    Recall: {recall}
    F1 Score: {f1}
    ''')

    return (precision, recall, f1)
