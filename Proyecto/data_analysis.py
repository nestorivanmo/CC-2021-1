import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sequential_df = pd.read_csv('data/output_linear.csv')
concurrent_df = pd.read_csv('data/output_parallel.csv')

sequential_pop_sizes = [5, 10, 15, 20, 30, 40, 60, 100]
concurrent_pop_sizes = [20, 30, 40, 60, 100, 400, 1000]
queens = [4, 5, 10, 15, 20, 25, 30, 40, 50]


def graph(test_df):
    """Graficando para cada reina tiempo promedio de ejecución sobre iteraciones promedio."""
    palette = sns.color_palette('magma', len(sequential_pop_sizes))
    for queen in queens:
        temp_df = test_df.loc[test_df.num_reinas == queen].copy()
        temp_df.drop(columns=['num_reinas'], inplace=True)
        g = sns.catplot(x='iter_avg', y='tiempo_avg', hue='tam_pob', data=temp_df, palette=palette)
        plt.ylabel('Número de iteraciones promedio')
        plt.xlabel('Tiempo promedio de ejecución [s]')
        plt.title('Tiempo promedio de ejecución contra número de iteraciones promedio')
        plt.tight_layout()
        plt.savefig('graphs/iter_tiempo_' +
                    str(queen) + '.png', bbox_inches='tight')
        plt.show()

    """Graficando para cada reina tamaño de población sobre tiempo promedio de ejecución"""
    palette = sns.color_palette('flare', len(sequential_pop_sizes))
    df = test_df.drop(columns=['iter_avg'])
    for queen in queens:
        temp_df = df.loc[df.num_reinas == queen]
        g = sns.catplot(x='tiempo_avg', y='tam_pob', col='num_reinas', data=temp_df, palette=palette)
        plt.ylabel('Tamaño de población')
        plt.xlabel('Tiempo promedio de ejecución [s]')
        plt.title('Tiempo promedio de ejecución contra población para ' + str(queen) + ' reinas')
        plt.savefig('graphs/tiempo_tam_' +
                    str(queen) + '.png', bbox_inches='tight')
        plt.show()

    """Graficando para cada reina tamaño de población sobre iteraciones promedio de ejecución"""
    df = test_df.drop(columns=['tiempo_avg'])
    for queen in queens:
        temp_df = df.loc[df.num_reinas == queen]
        g = sns.catplot(x='iter_avg', y='tam_pob', col='num_reinas', data=temp_df, palette=palette)
        plt.ylabel('Tamaño de población')
        plt.xlabel('Iteraciones promedio')
        plt.title('Iteraciones promedio contra población para ' + str(queen) + ' reinas')
        plt.savefig('graphs/iter_tam_' +
                    str(queen) + '.png', bbox_inches='tight')
        plt.show()

    palette = sns.color_palette('mako', len(queens))
    temp_df = test_df.drop(columns=['iter_avg'])
    g = sns.lineplot(x='tam_pob', y='tiempo_avg', data=temp_df, hue='num_reinas', style='num_reinas', markers=True,
                     dashes=False, palette=palette)
    plt.xlabel('Tamaño de población')
    plt.ylabel('Tiempo promedio de ejecución [s]')
    plt.title('Tiempo promedio de ejecución contra población')
    plt.savefig('graphs/tam_tiempo.png', bbox_inches='tight')
    plt.show()

    temp_df['tiempo_avg'] = np.log(temp_df['tiempo_avg'])
    temp_df['tam_pob'] = np.log(temp_df['tam_pob'])
    g = sns.lineplot(x='tam_pob', y='tiempo_avg', data=temp_df, hue='num_reinas', style='num_reinas', markers=True,
                     dashes=False, palette=palette)
    plt.xlabel('Tamaño de población')
    plt.ylabel('Tiempo promedio de ejecución [s]')
    plt.title('Tiempo promedio de ejecución contra población — log')
    plt.savefig('graphs/tam_tiempo_log.png', bbox_inches='tight')
    plt.show()

    temp_df = test_df.drop(columns=['tiempo_avg'])
    h = sns.lineplot(x='tam_pob', y='iter_avg', data=temp_df, hue='num_reinas', style='num_reinas', markers=True,
                     dashes=False, palette=palette)
    plt.xlabel('Tamaño de población')
    plt.ylabel('Número de iteraciones promedio')
    plt.title('Número de iteraciones promedio contra población')
    plt.savefig('graphs/tam_iter.png', bbox_inches='tight')
    plt.show()

    temp_df['iter_avg'] = np.log(temp_df['iter_avg'])
    temp_df['tam_pob'] = np.log(temp_df['tam_pob'])
    h = sns.lineplot(x='tam_pob', y='iter_avg', data=temp_df, hue='num_reinas', style='num_reinas', markers=True,
                     dashes=False, palette=palette)
    plt.xlabel('Tamaño de población')
    plt.ylabel('Número de iteraciones promedio')
    plt.title('Número de iteraciones promedio contra población — log')
    plt.savefig('graphs/tam_iter_log.png', bbox_inches='tight')
    plt.show()


def speedup(seq_df, con_df, seq_pop_sizes, con_pop_sizes):
    common_pop = list(set(sequential_pop_sizes) & set(concurrent_pop_sizes))
    common_pop.sort()
    df = pd.merge(seq_df, con_df, on=['num_reinas', 'tam_pob'],)
    df.columns = ['num_reinas', 'tam_pob', 'tiempo_avg_seq', 'iter_avg_seq', 'tiempo_avg_con', 'iter_avg_con']
    df['speedup'] = df.tiempo_avg_seq / df.tiempo_avg_con
    sns.lineplot(x='num_reinas', y='speedup', data=df)
    plt.xlabel('Número de reinas')
    plt.ylabel('Speedup')
    plt.title('Speedup con base en el número de reinas')
    plt.savefig('graphs/speedup_lin.png')
    plt.show()
    sns.lineplot(x='num_reinas', y='speedup', hue='tam_pob', data=df, palette=sns.color_palette('magma', 5))
    plt.xlabel('Número de reinas')
    plt.ylabel('Speedup')
    plt.title('Speedup con base en el número de reinas y tamaño de población')
    plt.savefig('graphs/speedup_seg.png', bbox_inches='tight')
    plt.show()
    df.to_csv('data/output.csv')
    return 1


def statistics(df=pd.read_csv('data/output.csv'), csv=False):
    stat_df, stat_seg_df = pd.DataFrame(), pd.DataFrame()
    stat_df['tiempo_avg_seq'] = df.tiempo_avg_seq.describe()
    stat_df['iter_avg_seq'] = df.iter_avg_seq.describe()
    stat_df['tiempo_avg_con'] = df.tiempo_avg_con.describe()
    stat_df['iter_avg_con'] = df.iter_avg_con.describe()
    stat_df['speedup'] = df.speedup.describe()
    stat_seg_df['num_queens'] = queens
    queen_time_seq, queen_time_con, queen_iter_seq, queen_iter_con  = [], [], [], []
    for queen in queens:
        queen_time_seq.append(np.average(df.tiempo_avg_seq[df.num_reinas == queen]))
        queen_time_con.append(np.average(df.tiempo_avg_con[df.num_reinas == queen]))
        queen_iter_seq.append(np.average(df.iter_avg_seq[df.num_reinas == queen]))
        queen_iter_con.append(np.average(df.iter_avg_con[df.num_reinas == queen]))
    stat_seg_df['tiempo_seq'] = queen_time_seq
    stat_seg_df['tiempo_con'] = queen_time_con
    stat_seg_df['iter_seq'] = queen_iter_seq
    stat_seg_df['iter_con'] = queen_iter_con
    if csv:
        stat_df.to_csv('data/stat.csv')
        stat_seg_df.to_csv('data/stat_seg.csv')


graph(concurrent_df)
speedup(sequential_df, concurrent_df, sequential_pop_sizes, concurrent_pop_sizes)
statistics()
