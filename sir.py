# Resources:
#   * https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model
#   * https://www.impan.pl/~slawek/sem/modele_epidemiologiczne.pdf
#   * https://www.maa.org/press/periodicals/loci/joma/the-sir-model-for-spread-of-disease-the-differential-equation-model
#   * https://www.lewuathe.com/covid-19-dynamics-with-sir-model.html
#   * https://ethz.ch/content/dam/ethz/special-interest/usys/ibz/theoreticalbiology/education/learningmaterials/701-1424-00L/sir.pdf

#  === Assumtions ===
# 1. We are working with closed population i.e. there is no immigration or emigration
# 2. We do not consider birth and natural death
# 3. Every infected person will recover at some point

import requests
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class SIR:
    """
    Class that represents the SIR Model:
        - S stands for susceptible
        - I stands for Infectiuos
        - R stands for Removed
    """

    def __init__(self, population, beta, gamma, initials):
        """
        Parameters
        ----------
        population: int
            The number of people in population
        beta: float
            The infectivity
        gamma: float
            The rate of recovery
        """
        self.population = population
        self.beta = beta
        self.gamma = gamma
        self.i0 = initials[0]
        self.s0 = population - self.i0
        self.r0 = initials[1]

    def formula(self, sir_arr, t):
        """
        Function that represents the forumla that calculates
        the equations
        """
        S, I, R = sir_arr
        dS = -(self.beta * I * S / self.population)
        dI = (self.beta * S * I / self.population) - self.gamma * I
        dR = self.gamma * I
        return np.array([dS, dI, dR])

    def sir(self, t_max):
        """
        Function that solves the differentaial equations
        """
        t = np.arange(0, t_max, 1)

        result = odeint(
            self.formula,
            np.array([self.s0, self.i0, self.r0]),
            t,
        )

        return result.T

    def plot(self, t_max, additional=np.array([])):
        """
        Function that shows the data on the plot

        Arguments
        ---------
        t_max: int
            the number of days of measurements
        additional: array
            the array with addintional data that will be placed on the plot
        """
        S, I, R = self.sir(t_max)
        fig, ax = plt.subplots()
        ax.plot(S, "r", label="Suspectible")
        ax.plot(I, "g", label="Infected")
        ax.plot(R, "b", label="Removed")
        if additional.size > 0:
            ax.plot(additional, "y", label="Actual")
        ax.legend()
        plt.show()


def get_data(link):
    """
    Function that downloads file from the link
        Parameters
        ----------
        link: string
            Link to the file with data
    """
    req = requests.get(link, allow_redirects=True)
    open("covid_data.csv", "wb").write(req.content)


def process_data():
    """
    Function puts gathered data in cache
    """
    get_data(
        "https://arcgis.com/sharing/rest/content/items/b03b454aed9b4154ba50df4ba9e1143b/data?"
    )
    csv_data = pd.read_csv(
        "covid_data.csv",
        encoding="ISO-8859-2",
        delimiter=";",
        header=None,
        usecols=[1, 2, 3, 5, 6, 7],
        skiprows=1,
    )
    return csv_data[2]


def get_actual():
    csv_data = process_data()
    data = np.array([])
    for value in csv_data:
        data = np.append(data, int(value.replace(" ", "")))
    return data


def compare():
    """
    Function that compares predicted data to the actual data.
    It does not work perfetcly but it is probably due to the constants that
    I could not work out to properly.

    It can be also caused by the unstable data of new infections in the Poland. Data was gathered in
    large period of time so the are fluctuations in the data.

    We have also used very simple model that does not include vital dynamics and imigration / emigration.
    """
    data = get_actual()
    sir = SIR(38386, 0.05, 0.01, [1, 0])
    sir.plot(266, data)


def main():
    compare()


if __name__ == "__main__":
    main()
