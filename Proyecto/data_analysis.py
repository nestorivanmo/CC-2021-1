import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

run_df = pd.read_csv('output.csv')
run_df.head()

pop_sizes = [5, 10, 15, 20, 30, 40, 60, 100]
queens = [4, 5, 10, 15, 20, 25, 30, 40, 50]

"""Graficando para cada reina tiempo promedio de ejecución sobre iteraciones promedio."""
palette = sns.color_palette('magma', 8)
for queen in queens:
    temp_df = run_df.loc[run_df.num_reinas == queen].copy()
    temp_df.drop(columns=['num_reinas'], inplace=True)
    g = sns.catplot(x='iter_avg', y='tiempo_avg', hue='tam_pob', data=temp_df, palette=palette)
    plt.xlabel('Número de iteraciones promedio')
    plt.xlabel('Tiempo promedio de ejecución [s]')
    plt.title('Tiempo promedio de ejecución contra número de iteraciones promedio')
    plt.savefig('graphs/iter_tiempo_' +
                str(queen) + '.png')
    plt.show()

"""Graficando para cada reina tamaño de población sobre tiempo promedio de ejecución"""
palette = sns.color_palette('flare', 8)
df = run_df.drop(columns=['iter_avg'])
for queen in queens:
    temp_df = df.loc[df.num_reinas == queen]
    g = sns.catplot(x='tiempo_avg', y='tam_pob', col='num_reinas', data=temp_df, palette=palette)
    plt.ylabel('Tamaño de población')
    plt.xlabel('Tiempo promedio de ejecución [s]')
    plt.title('Tiempo promedio de ejecución contra población para ' + str(queen) + ' reinas')
    plt.savefig('graphs/tiempo_tam_' +
                str(queen) + '.png')
    plt.show()

"""Graficando para cada reina tamaño de población sobre iteraciones promedio de ejecución"""
df = run_df.drop(columns=['tiempo_avg'])
for queen in queens:
    temp_df = df.loc[df.num_reinas == queen]
    g = sns.catplot(x='iter_avg', y='tam_pob', col='num_reinas', data=temp_df, palette=palette)
    plt.ylabel('Tamaño de población')
    plt.xlabel('Iteraciones promedio')
    plt.title('Iteraciones promedio contra población para ' + str(queen) + ' reinas')
    plt.savefig('graphs/iter_tam_' +
                str(queen) + '.png')
    plt.show()



palette = sns.color_palette('mako', 9)
temp_df = run_df.drop(columns=['iter_avg'])
g = sns.lineplot(x='tam_pob', y='tiempo_avg', data=temp_df, hue='num_reinas', style='num_reinas', markers=True,
                 dashes=False, palette=palette)
plt.xlabel('Tamaño de población')
plt.ylabel('Tiempo promedio de ejecución [s]')
plt.title('Tiempo promedio de ejecución contra población')
plt.savefig('graphs/tam_tiempo.png')
plt.show()

temp_df['tiempo_avg'] = np.log(temp_df['tiempo_avg'])
temp_df['tam_pob'] = np.log(temp_df['tam_pob'])
g = sns.lineplot(x='tam_pob', y='tiempo_avg', data=temp_df, hue='num_reinas', style='num_reinas', markers=True,
                 dashes=False, palette=palette)
plt.xlabel('Tamaño de población')
plt.ylabel('Tiempo promedio de ejecución [s]')
plt.title('Tiempo promedio de ejecución contra población — log')
plt.savefig('graphs/tam_tiempo_log.png')
plt.show()

temp_df = run_df.drop(columns=['tiempo_avg'])
h = sns.lineplot(x='tam_pob', y='iter_avg', data=temp_df, hue='num_reinas', style='num_reinas', markers=True,
                 dashes=False, palette=palette)
plt.xlabel('Tamaño de población')
plt.ylabel('Número de iteraciones promedio')
plt.title('Número de iteraciones promedio contra población')
plt.savefig('graphs/tam_iter.png')
plt.show()

temp_df['iter_avg'] = np.log(temp_df['iter_avg'])
temp_df['tam_pob'] = np.log(temp_df['tam_pob'])
h = sns.lineplot(x='tam_pob', y='iter_avg', data=temp_df, hue='num_reinas', style='num_reinas', markers=True,
                 dashes=False, palette=palette)
plt.xlabel('Tamaño de población')
plt.ylabel('Número de iteraciones promedio')
plt.title('Número de iteraciones promedio contra población — log')
plt.savefig('graphs/tam_iter_log.png')
plt.show()
