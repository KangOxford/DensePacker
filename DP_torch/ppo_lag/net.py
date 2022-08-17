import keras.backend as K
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.utils import plot_model

import yaml

with open('config.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    seed = cfg['setup']['seed']
    ymlfile.close()


tf.random.set_seed(seed)

"""DNN builder script

This manages the DNN creation and printing for the agent
"""


class DeepNetwork:
    """
    Class for the DNN creation
    """

    @staticmethod  
    def build(env, params, actor=False, name='model'):
        """Gets the DNN architecture and build it

        Args:
            env (gym): the gym env to take agent I/O
            params (dict): n° and size of hidden layers, print model for debug
            actor (bool): wether to build the actor or the critic
            name (str): file name for the model

        Returns:
            model: the uncompiled DNN Model
        """

        input_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]

        h_layers = params['h_layers']
        h_size = params['h_size']

        state_input = layers.Input(shape=(input_size,), name='input_layer')
        h = state_input
        for i in range(h_layers):
            h = layers.Dense(h_size, activation='tanh', name='hidden_' + str(i))(h)

        if actor:
            y = layers.Dense(action_size, activation=None, name='output_layer')(h)  
        else:
            y = layers.Dense(1, activation=None, name='critic_output_layer')(h)
   
        model = Model(inputs=state_input, outputs=y)

        # PNG with the architecture and summary
        if params['print_model']:
            plot_model(model, to_file=name + '.png', show_shapes=True)    
            model.summary()

        return model

    @staticmethod  
    def print_weights(model):
        """Gets the model and print its weights layer by layer

        Args:
            model (Model): model to print
           
        Returns:
            None
        """

        model.summary()
        print("Configuration: " + str(model.get_config()))
        for layer in model.layers:
            print(layer.name)
            print(layer.get_weights())  

