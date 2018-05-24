# Project: Perceptron Simple

__author__ = "Alejandro Sánchez Yalí"
__copyright__ = "Copyright 2009, Planet Earth"
__credits__ = "Alejandro Sánchez Yalí"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Alejandro Sánchez Yalí"
__email__ = "asanchezyali@gmail.com"
__status__ = "Production"
__date__ = "lun 30 abr 2018 15:22:51 -05"

""" Perceptron.

In this module, a simple neural network is programmed and implemented in a
perceptron. The classes implemented are Neuron and Perceptron.

Example:
   To instantiate the Neuron and Perceptron classes, write:
   
   >>> Neuron(np.array([1, 1, 1]))
   >>> Perceptron(n_receptors=5)
   
"""

import numpy as np
import math
import pickle
from itertools import product
import tkinter as tk
from typing import Callable


class Neuron(object):
    """
    In this class, the basic structure of a simple artificial neuron
    system is defined.

    Attributes:
        n_weight (np.array): It is the amount of synaptic weights.
        self.synaptic_weight (np.array): It is a vector with the synaptic
        weights by default it is full of ones.
        self.memory (pickle): It is the memory where new knowledge is stored.
    """

    def __init__(self, dendrite: np.array, sensitivity:
                 int=2, dilatation: float=0.5) -> object:

        """
        The function __init__ is a method for initializing the class Neuron.
       
        Args:
            dendrite: It is a np.array for example:

            >>> dendrite = np.array([1, 1, -1, 1, 1, 1])
            >>> Neuron(dendrite)
            
            sensitivity: It is a int number that defines the sensitivity of
            the neuron. This number real is a positive integer.
            
            dilatation: It is a real number that defines the dilatation for 
            the active function.This number real is between 0 and 1.
        """

        sensitivity = '9' * int(sensitivity)
        self.sensitivity = 1 - 100 / int(sensitivity)

        # Attributes of class.
        self.dilatation = 1/dilatation
        self.n_weight = len(dendrite)
        self.synaptic_weight = np.ones(self.n_weight)

        # We define the memory of neuron.
        pickle.dump([], open('memory.mem', 'wb'))
        self.memory = pickle.load(open('memory.mem', 'rb'))

    def soma(self, dendrite: np.array, potential_function: Callable=np.dot,
             active_function: Callable=math.tanh) -> float:
        """
        The soma is the computation center, here the axon activating signal is
        calculated.

        Args:
            dendrite: It is a np.array for example:

            >>> dendrite = np.array([1, 1, -1, 1, 1, 1])
            >>> Neuron(dendrite)
            potential_function: It is a function of two real variables.
            active_function: It is a function of a real variables.

        Returns:
            float: It is a number with axon activation signal.

        """

        potential = potential_function(dendrite, self.synaptic_weight)
        potential = potential / self.dilatation
        order = active_function(potential)
        return order

    def learn(self, dendrite: np.array) -> None:
        """
        The learn function allows the neuron to learn new patterns.
        Args:
            dendrite: It is a np.array for example:

            >>> dendrite = np.array([1, 1, -1, 1, 1, 1])
            >>> Neuron(dendrite)

        Returns:
            None
        """

        self.memory.append(dendrite)
        self.memory.reverse()
        print('Signal learned')

    def forget(self) -> None:
        """
        This function allows you to erase the memory

        Returns:
            None
        """

        self.memory = []
        pickle.dump(self.memory, open("memory.mem", 'wb'))
        print('Memory deleted')

    def axon(self, dendrite: np.array) -> np.array:

        """
        This function decides whether the axon emits the recognition signal
        or not
        :param dendrite: It is a np.array for example:

            >>> dendrite = np.array([1, 1, -1, 1, 1, 1])
            >>> Neuron(dendrite)

        :return:
            predict_signal: It is a np.array with the signal that the axon
            emits.
        """

        candidate_signals = {}

        # Identifying all the possible signals that the axon can emit.
        for mem in self.memory:
            self.synaptic_weight = mem
            weight = self.soma(dendrite)
            if weight > self.sensitivity:
                candidate_signals[weight] = mem

        if not candidate_signals:
            predict_signal = None
        else:
            # Choosing the best signal
            predict_weight = max(candidate_signals.keys())
            print(predict_weight)
            predict_signal = candidate_signals[predict_weight]
        return predict_signal


class Perceptron(object):
    """
    In the perceptron class, the graphical interface is defined and controlled
    by an instance of the neuron class.

    Attributes:
        button_size (int): It is a number to define the button size.
        signal (np.array): It is a vector with n entries full with -1.
        neuron (Neuron): It is an instance of the neuron class.
    """

    def __init__(self, sqrt_n_receptors: int=5, sensitivity: int=17, focus:
                 float=1) -> object:
        """
        The function __init__ is a method for initializing the class Neuron.
        Args:
            sqrt_n_receptors: It receives an integer n and creates n * n
            receivers.
            sensitivity: It is a int number that defines the sensitivity of
            the neuron. This number real is a positive integer.
            focus: It is a real number that defines the dilatation for
            the active function. This number real is between 0 and 1.
        """

        self.sqrt_n_receptors = sqrt_n_receptors
        self.sensitivity = sensitivity
        self.focus = focus

        # Attributes of class.
        self.signal = np.array([-1 for _ in range(self.sqrt_n_receptors ** 2)])
        self.neuron = Neuron(self.signal, self.sensitivity, self.focus)

        # Main window with buttons.
        main_window = tk.Tk()
        main_window.title('Perceptron')
        main_window.resizable(width=False, height=False)

        # We define all buttons for the main windows.
        n_row = range(sqrt_n_receptors)
        self.buttons = [[None]*sqrt_n_receptors for _ in n_row]
        self.buttons = np.array(self.buttons)
        self.values = np.zeros((self.sqrt_n_receptors, self.sqrt_n_receptors))
        self.coordinates = {}

        kwargs = dict(text=' ', bg='gray', relief='flat', width=2, height=2)

        for row, col in product(n_row, n_row):
            self.buttons[row, col] = tk.Button(main_window, **kwargs)
            self.buttons[row, col].grid(row=row, column=col, padx=2, pady=2)
            self.coordinates[self.buttons[row, col]] = [row, col]

        # We detect the events of each button.
        for button in self.buttons.flat:
            button.bind("<Button-1>", self.button_pressed)

        # Second window with buttons.
        second_window = tk.Tk()
        second_window.title('')
        second_window.geometry("110x110")
        second_window.resizable(width=False, height=False)

        # We define all buttons for the second windows.
        kwargs_memorize = dict(text='learn', command=self.memorize, width=455)
        button_memorize = tk.Button(second_window, **kwargs_memorize)
        kwargs_analyze = dict(text='Analyze', command=self.id_signal, width=455)
        button_analyze = tk.Button(second_window, **kwargs_analyze)
        kwargs_reset = dict(text='Reset', command=self.reset, width=455)
        button_reset = tk.Button(second_window, **kwargs_reset)
        kwargs_forget = dict(text='Forget', command=self.forget, width=455)
        button_forget = tk.Button(second_window, **kwargs_forget)

        # We store in a pack.
        button_memorize.pack()
        button_analyze.pack()
        button_reset.pack()
        button_forget.pack()

        main_window.mainloop()
        second_window.mainloop()

    def button_pressed(self, event: Callable) -> None:
        """
        This function modifies the color of the buttons in the main window. It
        also changes the signal to learn or identify.

        Args:
            event: It is a register of event in the main window.

        Returns:
            None
        """

        # We identify the coordinates of event.
        row, col = self.coordinates[event.widget]
        n = self.sqrt_n_receptors * row + col

        if self.signal[n] == -1:
            self.signal[n] = 1
            self.buttons[row][col]['bg'] = '#5BADFF'
        else:
            self.signal[n] = -1
            self.buttons[row][col]['bg'] = 'gray'

    def id_signal(self) -> None:
        """
        This function is to identify the signal. To do this, method axon of the
        Neuron class is used.

        Returns:
            None.
        """

        signal = self.neuron.axon(self.signal)
        try:
            n = len(signal)
        except TypeError:
            print('I do not know is this')
        else:
            for i in range(n):
                if signal[i] == 1:
                    row = i // self.sqrt_n_receptors
                    col = i % self.sqrt_n_receptors
                    self.buttons[row][col]['bg'] = '#01D826'

    def reset(self) -> None:
        """
        This function allows you to reset the main window.

        Returns:
            None.
        """

        self.signal = -1 * np.ones(len(self.signal))
        size = range(self.sqrt_n_receptors)
        for row, col in product(size, size):
            self.buttons[row][col]['bg'] = 'gray'

    def memorize(self) -> None:
        """
        The memorize function allows the perceptron to learn new patterns. To do
        this, method learn of the Neuron class is used.

        Returns:
            None.
        """

        self.neuron.learn(self.signal)
        self.reset()

    def forget(self) -> None:
        """
        The forget functions is to delete the memory in control neuron.

        Returns:
            None.
        """
        self.neuron.forget()


# We instance the main program.
if __name__ == "__main__":
    Perceptron()

