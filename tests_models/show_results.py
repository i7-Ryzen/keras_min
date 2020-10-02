import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from texttable import Texttable
from statistics import mean



def dist_plot(dictio, model_name, test_name):
    for name in dictio.keys():
        # Draw the density plot
        sns.distplot(dictio[name], hist=True, kde=True,
                     kde_kws={'linewidth': 3},
                     label= name)

    # Plot formatting
    plt.legend(prop={'size': 7})
    plt.title(model_name + ' : density_plot runtime  for ' + test_name, loc='center', wrap=True)
    plt.xlabel('runtime (s)')
    plt.ylabel('Density')
    plt.savefig(Path(__file__).parent.absolute() / "plots" /  ("density_plot_" + model_name +"_" + test_name))
    plt.show()


def line_plot(dictio, model_name, test_name):
    for name in dictio.keys():
        # Draw the density plot
        sns.lineplot(x = np.arange(len(dictio[name])) ,y = dictio[name], label=name)

    # Plot formatting
    plt.legend(prop={'size': 7})
    plt.title(model_name + ' : line_plote runtime  for ' + test_name, loc='center', wrap=True)
    plt.xlabel('iterations')
    plt.ylabel('runtime (s)')
    plt.savefig(Path(__file__).parent.absolute() / "plots" / ("line_plot_" + model_name +"_" + test_name))
    plt.show()



def show_results(keras_lists, kerasmin_lists, model_name):
    """
    :param keras_lists: is a tuple of (first_predictions_list, second_predictions_list, load_model_keras, output_keras)
    :param kerasmin_lists: is a tuple of (first_predictions_list), second_predictions_list, load_model_kerasmin, output_kerasmin)
    :return: printing the table of results and showing the plots
    """
    # keras
    first_predictions_list_keras  = keras_lists[0]
    second_predictions_list_keras  = keras_lists[1]
    load_model_keras = keras_lists[2]
    output_keras = keras_lists[3]

    # kerasmin
    first_predictions_list_kerasmin  = kerasmin_lists[0]
    second_predictions_list_kerasmin  = kerasmin_lists[1]
    load_model_kerasmin = kerasmin_lists[2]
    output_kerasmin = kerasmin_lists[3]

    # print table of results
    t = Texttable()
    t.add_rows([['Name of model: ' + model_name, 'first_predictions_time_average', 'second_predictions_time_average', 'load_model_time_average','prediction_output' ],
                ['Keras', mean(first_predictions_list_keras), mean(second_predictions_list_keras) ,mean(load_model_keras), output_keras[0]],
                ['Kerasmin', mean(first_predictions_list_kerasmin), mean(second_predictions_list_kerasmin) ,mean(load_model_kerasmin), output_kerasmin[0]]])
    print(t.draw())

    f = open(Path(__file__).parent.absolute() / "tables_results" /(model_name + ".txt"), "w")
    f.write(t.draw())
    f.close()


    # show plots
    dictio_first_deployment = {"Numpy":first_predictions_list_kerasmin, "keras":first_predictions_list_keras}
    dictio_second_deployment = {"Numpy": second_predictions_list_kerasmin, "keras": second_predictions_list_keras}
    dictio_loading_model = {"Numpy": load_model_kerasmin, "keras": load_model_keras}

    # first_deployment
    dist_plot(dictio_first_deployment, model_name, "first_deployment")
    line_plot(dictio_first_deployment, model_name, "first_deployment")

    # second_deployment
    dist_plot(dictio_second_deployment, model_name, "second_deployment")
    line_plot(dictio_second_deployment, model_name, "second_deployment")

    # loading_model
    dist_plot(dictio_loading_model, model_name, "loading_model")
    line_plot(dictio_loading_model, model_name, "loading_model")





