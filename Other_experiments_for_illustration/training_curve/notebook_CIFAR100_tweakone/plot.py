import matplotlib.pyplot as plt
import pickle
import numpy as np


with open('lists.pkl', 'rb') as f:
    loaded_list_obj = pickle.load(f)
    loaded_list_mse = pickle.load(f)


def plot(obj_fun_list, MSE_list):
    print(len(obj_fun_list), len(MSE_list))
    num_iter = len(obj_fun_list)
    MSE_plot = []
    obj_plot = []
    iter_plot = []
    for i in range(30):
        iter_plot.append(i*500)
        MSE_plot.append(loaded_list_mse[i*500])
        obj_plot.append(loaded_list_obj[i*500])
    #plt.figure(1)
    plt.figure(figsize=(6, 6))
    plt.plot(np.array(iter_plot), MSE_plot)
    plt.xlabel("Number of Iterations", fontsize = 22)
    plt.ylabel("Mean Squared Error", fontsize = 22)
    plt.grid(which='major', linestyle='-', linewidth='0.5')
    plt.savefig("./MSE_curve.pdf")
    plt.show()

    #plt.figure(2)
    plt.figure(figsize=(6, 6))
    plt.plot(np.array(iter_plot), obj_plot)
    plt.xlabel("Number of Iterations", fontsize = 22)
    plt.ylabel("Objective Function", fontsize = 22)
    plt.grid(which='major', linestyle='-', linewidth='0.5')
    plt.savefig("./Obj_curve.pdf")
    plt.show()

plot(loaded_list_obj, loaded_list_mse)