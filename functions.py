import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from graphviz import Digraph
from datetime import datetime, timedelta

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


class Transaction():
    
    def __init__(self, DataFrame, col_transact_name, col_time_name, col_identifier_name):
        self.df = DataFrame.copy()
        self.tr_col_name = col_transact_name
        self.time_col_name = col_time_name
        self.t_date = pd.to_datetime('2060-01-01 00:00:00.00')
        self.df.rename(columns={col_identifier_name:"case_name"}, inplace=True)
        self.id_col_name = "case_name"
        if type(self.df[self.time_col_name][self.df.index[0]]) == str:
            self.df[self.time_col_name] = pd.to_datetime(self.df[self.time_col_name])
        
    def group_case_by_time(self):
        self.df = self.df.set_index([self.id_col_name, self.time_col_name]).sort_index()
        self.df = self.df.reset_index()
    
    def shift_process_col(self):
        self.df['name_2'] = self.df.groupby(self.id_col_name)[self.tr_col_name].shift(periods=-1, fill_value = 'Конец лога')
    
    def shift_time_col(self):
        self.df['time_2'] = self.df.groupby(self.id_col_name)[self.time_col_name].shift(periods=-1, fill_value = self.t_date)
    
    def time_diff(cls):
        if 'time_2' not in cls.df.columns:
            cls.shift_time_col()
        cls.df['time_diff'] = cls.df['time_2'] - cls.df[cls.time_col_name]
        cls.df['time_diff'] = cls.df['time_diff'].dt.total_seconds()
        cls.df.loc[cls.df['time_diff'] > 200000000, 'time_diff'] = 0
    
    def transact(cls):
        if 'name_2' not in cls.df.columns:
            cls.shift_process_col()
        cls.df['transact'] = cls.df[cls.tr_col_name].astype('str')+'-->'+cls.df['name_2'].astype('str')
    
    def preparing(cls):
        cls.group_case_by_time()
        cls.shift_process_col()
        cls.shift_time_col()
        cls.time_diff()
        cls.transact()
        return cls.df[["case_name", "transact", "time_diff"]].reset_index(drop=True)




class Prepare(Transaction):
    
    def __init__(self, DataFrame, col_transact_name, col_time_name, col_identifier_name):
        super().__init__(DataFrame, col_transact_name, col_time_name, col_identifier_name)
    
    def add_init_process(cls):
        time_start = cls.df[cls.time_col_name].min() - timedelta(seconds=1)
        concept_name = 'Начало лога'
        cs_name = cls.df[cls.id_col_name].unique()
        result = pd.DataFrame({cls.id_col_name: cs_name,
                         cls.tr_col_name: concept_name,
                         cls.time_col_name: time_start})
        cls.df = pd.concat([result, cls.df])
        
    def get_result(cls):
        cls.add_init_process()
        cls.df = cls.preparing()
        cls.df.loc[cls.df['transact'].str.contains('Начало лога'), 'time_diff'] = 0
        return cls.df


class KMeans_Clusterization():
    def __init__(self, model, DataFrame):
        self.model_ = model
        self.df = DataFrame
        self.inertia = []
    
    def elbow_method(self):
        for k in np.array(range(1,10)):
            self.model_.set_params(n_clusters=k)
            kmns = self.model_.fit(self.df.values)
            self.inertia.append(kmns.inertia_)
        
    def draw_elbow_method_plot(cls):
        if "KMeans" not in str(cls.model_.__str__):
            return "Данный метод работает для KMeans - алгоритма"
        else:
            cls.elbow_method()
            plt.figure(figsize=(12,8))
            plt.plot(np.array(range(1,10)), np.array(cls.inertia), 'rs-');
            plt.xlabel('k')
            plt.ylabel('$J(c_k)$')
            plt.show();
        
    def clustering(self, num_clusters):
        self.model_.set_params(n_clusters=num_clusters)
        self.df["clusters"] = self.model_.fit_predict(self.df.values) 
        return self.df.reset_index()    
    

    
class DBSCAN_Clusterization():
    def __init__(self, pivot_tabl):
        self.p_table = pivot_tabl
        self.pca_res = None
        self.dist = None
        self.ind = None
    
    def preparing(self):
        sc = StandardScaler()
        X_scaled = sc.fit_transform(self.p_table)
        pca = PCA(n_components=self.p_table.columns.shape[0])
        self.pca_res = pca.fit_transform(X_scaled)
    
    def calculate_distances(cls):
        cls.preparing()
        n_ngbr = NearestNeighbors(n_neighbors=2)
        neighbors = n_ngbr.fit(cls.pca_res)
        cls.dist, cls.ind = neighbors.kneighbors(cls.pca_res)
        
    def epsilon_optimal_graph(cls, ranges=None):
        if cls.dist is None:
            cls.calculate_distances()
        plt.figure(figsize=(8,8))
        plt.plot(np.sort(cls.dist[:,1], axis=0))
        plt.xlabel("Points")
        plt.ylabel("Distance")
        if ranges is not None:
            plt.axis(ranges)
            for i in range(1, ranges[-1]):
                plt.axhline(y=i, color='r', linestyle='-', linewidth=0.1)
        
    def clustering(cls, eps_val, min_samples):
        if cls.pca_res is None:
            cls.preparing()
        model = DBSCAN(eps=eps_val, n_jobs=-1, min_samples=min_samples)
        model.fit(cls.pca_res)
        cls.p_table["clusters"] = model.labels_
        cls.p_table.reset_index(inplace=True)
        print(f"Количество кластеров равно {len(np.unique(model.labels_))}")
        return cls.p_table    
    

class Select_cluster():
    
    def __init__(self, DataFrame, num_cluster):
        self.df = DataFrame
        self.n_cluster = num_cluster
        
    def select(self):
        ids = self.df[self.df["clusters"] == self.n_cluster]["case_name"].unique()
        for_draw = self.df[self.df["case_name"].isin(ids)]
        print(f"Уникальных идентификаторов = {for_draw['case_name'].unique().shape[0]}")
        return for_draw


class Frequency_graph():
    """
    DataFrame - с 3 столбцами ['case_name', 'transact', 'time_diff'].
    count_treshold - принимает на вход:
        - string: "All" (отрисовываются ребра любой частотности)
        - integer: 0, 10, 123,... (отрисовываются ребра с частотой до или с заданного диапазона,
                                    указывается совместно с параметром less_or_more) 
    less_or_more - принимает на вход:
        - string: "<" или ">" (указывается для задания необходимого порога, если параметр 
                                count_treshold принимает значение integer)
    Функция возвращает граф, где цифрами обозначена частота перехода между двумя событиями.
                             Чем ярче и толще линия, тем чаще встречается данный переход.
    """
    def __init__(self, DataFrame, filename = None):
        self.df = DataFrame
        self.name = filename
        self.counts = None
        self.graph = None
    
    def check_colon(self):
        name_processes = tuple(self.df['transact'].unique())
        counts = [1 for i in name_processes if ':' in i]
        if len(counts) != 0:
            print("""В именах событий содержится двоеточие, данный вывод некорректен, 
                     перед отрисовкой графа необходимо заменить двоеточие на тире""")
    
    def graph_init(self):
        if self.name is None:
            curr_time = datetime.now().strftime("%H:%M:%S").replace(':', '-')
            self.graph = Digraph('finite_state_machine', filename=f'Frequency_graph_{curr_time}')
        else:
            self.graph = Digraph('finite_state_machine', filename=self.name)
        self.graph.attr(rankdir='T', size='8,5')
        self.graph.attr('node', shape='box', style='filled', color='deepskyblue')
        self.graph.node('Начало лога', shape='doublecircle', color='deepskyblue1')
        self.graph.node('Конец лога', shape='doublecircle', color='brown3')
        self.graph.attr('node', shape='box', color='lightblue')
    
    
    def stat_calculate(self, count_treshold, less_or_more):
        if count_treshold == 'All':
            self.counts = pd.DataFrame(self.df['transact'].value_counts()).reset_index()
        else:
            try:
                if less_or_more == "<":
                    counts = pd.DataFrame(self.df['transact'].value_counts()).reset_index()
                    self.counts = counts[counts['transact'] < count_treshold]
                elif less_or_more == ">":
                    counts = pd.DataFrame(self.df['transact'].value_counts()).reset_index()
                    self.counts = counts[counts['transact'] > count_treshold]
                else:
                    return 'Недопустимое значение для параметра less_or_more или неправильно указано значение All'
            except Exception as e:
                print(e)
                print("""Параметр count_treshold принимает либо значение All, либо  должно иметь тип integer,\
                совместно с параметром less_or_more, который принимает одно из следующих значений: "<" или ">" """)
    
    @staticmethod
    def change_color_freq(count_transact, stat):
        if count_transact <= stat[0]:
            color = 'bisque3'#'brown'
        elif (count_transact > stat[0]) and (count_transact <= stat[1]):
            color = 'bisque3'#'coral1'
        elif (count_transact > stat[1]) and (count_transact <= stat[2]):
            color = 'bisque3'#'goldenrod'
        elif (count_transact > stat[2]) and (count_transact <= stat[3]):
            color = 'deepskyblue1'
        elif count_transact > stat[3]:
            color = 'cyan'
        return color
    
    @staticmethod
    def change_width_freq(count_transact, stat):
        if count_transact <= stat[0]:
            width = '1'
        elif (count_transact > stat[0]) and (count_transact <= stat[1]):
            width = '1'#'2'
        elif (count_transact > stat[1]) and (count_transact <= stat[2]):
            width = '1'#'3'
        elif (count_transact > stat[2]) and (count_transact <= stat[3]):
            width = '4'
        elif count_transact > stat[3]:
            width = '5'
        return width
    
    @staticmethod
    def get_stat_freq(freq_to_int):
        percent_25 = int(np.percentile(freq_to_int, 25))
        percent_50 = int(np.percentile(freq_to_int, 50))
        percent_75 = int(np.percentile(freq_to_int, 75))
        percent_95 = int(np.percentile(freq_to_int, 95))
        stat_percent = [percent_25, percent_50, percent_75, percent_95]
        return stat_percent
    
    def draw_freq(cls, count_treshold = 'All', less_or_more = None):
        cls.stat_calculate(count_treshold, less_or_more)
        cls.graph_init()
        cls.check_colon()
        
        cls.counts.columns = ['transact', 'counts']
        transact = cls.counts['transact'].values
        countss = cls.counts['counts'].values
        
        for c in range(len(transact)):
            stat_percent = get_stat_freq(countss)
            tr = transact[c]
            count = int(countss[c])
            start = tr.split('-->')[0]
            end = tr.split('-->')[1]
            cls.graph.edge('{0}'.format(start), 
                   '{0}'.format(end), 
                   label='{0}'.format(count), 
                   arrowhead='vee', 
                   penwidth=change_width_freq(count, stat_percent), 
                   color = change_color_freq(count, stat_percent), 
                   fontcolor=change_color_freq(count, stat_percent))
        cls.graph.view()

        
class Performance_graph(Frequency_graph):
    """
    DataFrame с 3 столбцами ['case_name', 'transact', 'time_diff'].
    time_treshold - временнОй порог для отрисовки:
        - string: "All" (отрисовка ребер любой длительности)
        - integer: 0, 10, 123,... (отрисовка ребер с событиями, между которыми время исполнения больше или 
                                   меньше заданного, указывается в часах, совместно с параметром less_or_more) 
    less_or_more - принимает на вход:
        - string: "<" или ">" (указывается для задания необходимого порога)
    type_value - тип значения времени:
        - string: "min","max", "median" (какой тип применить для расчета времени)
    Функция возвращает граф, где на ребрах обозначены время исполнения перехода между двумя событиями.
                             Чем темнее и толще линия, тем дольше исполняется данный переход.
    """
    def __init__(self, DataFrame, filename = None):
        super().__init__(DataFrame, filename = None)
        self.med_time = None
    
    def time_calculate(self, time_treshold, type_value, less_or_more):
        if time_treshold == 'All':
            self.med_time = pd.DataFrame(self.df.groupby('transact')['time_diff'].agg(['min', 
                                                                                      'max', 
                                                                                      'median'])).reset_index()
        else:
            try:
                if less_or_more == "<":
                    med_time = pd.DataFrame(self.df.groupby('transact')['time_diff'].agg(['min', 
                                                                                              'max', 
                                                                                              'median'])).reset_index()
                    self.med_time = med_time[med_time[type_value] < time_treshold*3600]
                elif less_or_more == ">":
                    med_time = pd.DataFrame(self.df.groupby('transact')['time_diff'].agg(['min', 
                                                                                              'max', 
                                                                                              'median'])).reset_index()
                    self.med_time = med_time[med_time[type_value] > time_treshold*3600]
                else:
                    return 'Недопустимое значение для параметра less_or_more или неправильно указано значение All'
            except Exception as e:
                print(e)
                print('Параметр time_treshold принимает либо значение All, либо  должно иметь тип integer,\
                совместно с параметром less_or_more, который принимает одно из следующих значений: "<" или ">"')
    
    @staticmethod
    def change_width(secs):
        if secs <= 1:
            width = '1'
        elif (secs > 1) and (secs <= 60):
            width = '2'
        elif (secs > 60) and (secs <= 3600):
            width = '3'
        elif (secs > 3600) and (secs <= 36000):
            width = '4'
        elif secs > 36000:
            width = '5'
        return width

    @staticmethod
    def change_color(secs):
        if secs <= 1:
            color = 'cyan'
        elif (secs > 1) and (secs <= 60):
            color = 'deepskyblue1'
        elif (secs > 60) and (secs <= 3600):
            color = 'goldenrod'
        elif (secs > 3600) and (secs <= 36000):
            color = 'coral1'
        elif secs > 36000:
            color = 'brown'
        return color
    
    @staticmethod
    def secondsToText(secs):
        if secs > 1:
            days = round(secs//86400)
            hours = round((secs - days*86400)//3600)
            minutes = round((secs - days*86400 - hours*3600)//60)
            seconds = round(secs - days*86400 - hours*3600 - minutes*60)
            result = ("{}d:".format(days) if days else "") + \
            ("{}h:".format(hours) if hours else "") + \
            ("{}m:".format(minutes) if minutes else "") + \
            ("{}s:".format(seconds))
        else:
            result = str(round(secs, 4)) +' '+ 'sec'
        return result
    
    def draw_perform(cls, time_treshold = 'All', type_value = 'median', less_or_more = '>'):
        cls.time_calculate(time_treshold, type_value, less_or_more)
        cls.graph_init()
        cls.check_colon()
        
        list_trans = cls.med_time['transact'].values
        times = cls.med_time['median'].values
        for c in range(len(list_trans)):
            tr = list_trans[c]
            time = float(times[c])
            start = tr.split('-->')[0]
            end = tr.split('-->')[1]
            cls.graph.edge('{0}'.format(start), '{0}'.format(end), 
                   label='{0}'.format(cls.secondsToText(time)), 
                   arrowhead='vee', 
                   penwidth=cls.change_width(time), 
                   color=cls.change_color(time),
                   fontcolor=cls.change_color(time))
        cls.graph.view()        
        

def draw_frequency_graph(df_prepared, name_file = None, count_treshold = 'All', less_or_more = None):
    
    """
    df_prepared - DataFrame, с 3 столбцами ['case_name', 'transact', 'time_diff'].
    count_treshold - принимает на вход:
        - string: "All" (отрисовываются ребра любой частотности)
        - integer: 0, 10, 123,... (отрисовываются ребра с частотой до или с заданного диапазона,
                                    указывается совместно с параметром less_or_more) 
    less_or_more - принимает на вход:
        - string: "<" или ">" (указывается для задания необходимого порога, если параметр 
                                count_treshold принимает значение integer)
    Функция возвращает граф, где цифрами обозначена частота перехода между двумя событиями.
                             Чем ярче и толще линия, тем чаще встречается данный переход.
    """
    if count_treshold == 'All':
        counts = pd.DataFrame(df_prepared['transact'].value_counts()).reset_index()#.sort_values(by='transact')
    else:
        try:
            if less_or_more == "<":
                counts = pd.DataFrame(df_prepared['transact'].value_counts()).reset_index()
                counts = counts[counts['transact'] < count_treshold]
            elif less_or_more == ">":
                counts = pd.DataFrame(df_prepared['transact'].value_counts()).reset_index()
                counts = counts[counts['transact'] > count_treshold]
            else:
                return 'Недопустимое значение для параметра less_or_more или неправильно указано значение All'
        except Exception as e:
            print(e)
            print('Параметр count_treshold принимает либо значение All, либо  должно иметь тип integer,\
            совместно с параметром less_or_more, который принимает одно из следующих значений: "<" или ">"')

    #if (name_file is not None) & ('.pdf' not in name_file):
        #name_file = name_file + '.pdf'        
            
    counts.columns = ['transact', 'counts']
    transact = counts['transact'].values
    countss = counts['counts'].values
    
    if name_file is None:
        curr_time = datetime.now().strftime("%H:%M:%S").replace(':', '-')
        f = Digraph('finite_state_machine', filename=f'Graphs/Frequency_graph_{curr_time}')
    else:
        f = Digraph('finite_state_machine', filename=name_file)
    
    f.attr(rankdir='T', size='8,5')
    
    f.attr('node', shape='box', style='filled', color='deepskyblue')
    f.node('Начало лога', shape='doublecircle', color='deepskyblue1')
    f.node('Конец лога', shape='doublecircle', color='brown3')
    f.attr('node', shape='box', color='lightblue')
    
    for c in range(len(transact)):
        stat_percent = get_stat_freq(countss)
        tr = transact[c]
        count = int(countss[c])
        start = tr.split('-->')[0]
        end = tr.split('-->')[1]

        f.edge('{0}'.format(start), 
               '{0}'.format(end), 
               label='{0}'.format(count), 
               arrowhead='vee', 
               penwidth=change_width_freq(count, stat_percent), 
               color = change_color_freq(count, stat_percent), 
               fontcolor=change_color_freq(count, stat_percent))
    
    print('Если всплывает много предупреждений, подсвеченных красным (данный вывод некорректен),\
    значит в имени события содержится двоеточие, которое нужно заменить на тире')
    f.view()        
        
        
def draw_performance_graph(df_prepared, name_file = None, time_treshold = "All", less_or_more = None, type_value = 'median'):
    """
    df_prepared - DataFrame с 3 столбцами ['case_name', 'transact', 'time_diff'].
    time_treshold - временнОй порог для отрисовки:
        - string: "All" (отрисовка ребер любой длительности)
        - integer: 0, 10, 123,... (отрисовка ребер с событиями, между которыми время исполнения больше или 
                                   меньше заданного, указывается в часах, совместно с параметром less_or_more) 
    less_or_more - принимает на вход:
        - string: "<" или ">" (указывается для задания необходимого порога)
    type_value - тип значения времени:
        - string: "min","max", "median" (какой тип применить для расчета времени)
    Функция возвращает граф, где на ребрах обозначены время исполнения перехода между двумя событиями.
                             Чем темнее и толще линия, тем дольше исполняется данный переход.
    """
    if time_treshold == 'All':
        med_time = pd.DataFrame(df_prepared.groupby('transact')['time_diff'].agg(['min', 
                                                                                  'max', 
                                                                                  'median'])).reset_index()
    else:
        try:
            if less_or_more == "<":
                med_time = pd.DataFrame(df_prepared.groupby('transact')['time_diff'].agg(['min', 
                                                                                          'max', 
                                                                                          'median'])).reset_index()
                med_time = med_time[med_time[type_value] < time_treshold*3600]
            elif less_or_more == ">":
                med_time = pd.DataFrame(df_prepared.groupby('transact')['time_diff'].agg(['min', 
                                                                                          'max', 
                                                                                          'median'])).reset_index()
                med_time = med_time[med_time[type_value] > time_treshold*3600]
            else:
                return 'Недопустимое значение для параметра less_or_more или неправильно указано значение All'
        except Exception as e:
            print(e)
            print('Параметр time_treshold принимает либо значение All, либо  должно иметь тип integer,\
            совместно с параметром less_or_more, который принимает одно из следующих значений: "<" или ">"')
    
    #if (name_file is not None) & ('.pdf' not in name_file):
        #name_file = name_file + '.pdf'
    
    list_trans = med_time['transact'].values
    times = med_time['median'].values
    
    
    if name_file is None:
        curr_time = datetime.now().strftime("%H:%M:%S").replace(':', '-')
        f = Digraph('finite_state_machine', filename=f'Performance_graph_{curr_time}')
    else:
        f = Digraph('finite_state_machine', filename=name_file)
    
    f.attr(rankdir='T', size='8,5')

    f.attr('node', shape='box', style='filled', color='deepskyblue')
    f.node('Начало лога', shape='doublecircle')
    f.node('Конец лога', shape='doublecircle', color='brown3')
    f.attr('node', shape='box', color='lightblue')

    for c in range(len(list_trans)):
        tr = list_trans[c]
        time = float(times[c])
        start = tr.split('-->')[0]
        end = tr.split('-->')[1]

        f.edge('{0}'.format(start), '{0}'.format(end), 
               label='{0}'.format(secondsToText(time)), 
               arrowhead='vee', 
               penwidth=change_width(time), 
               color = change_color(time),
               fontcolor=change_color(time))
    
    print('Если всплывает много предупреждений, подсвеченных красным,\
    значит в имени события содержится двоеточие, которое нужно заменить на тире')
    f.view()
    
    

# def get_all_process_motion(df, col_transact):
#     """
#     Input: df with 3 columns - [case_name, concept:name (col_transact), time:timestamp]
#     Output: df with
#     """
#     t_date = pd.to_datetime('2060-01-01 00:00:00.00') 
    
#     sort_grup_1 = df.groupby('case_name').apply(lambda x: x.sort_values('time:timestamp'))
# #     sort_grup_1 = df.copy()
#     print('Sort by time at {0}'.format(str(datetime.now().time())))
    
#     sort_grup_1['concept:name:2'] = pd.DataFrame(sort_grup_1.groupby('case_name').apply(lambda x: \
#                          x[col_transact].shift(periods=-1).fillna('Конец лога')))[col_transact].values
#     print('Shift processes at {0}'.format(str(datetime.now().time())))
    
#     sort_grup_1['time:timestamp:2'] = pd.DataFrame(sort_grup_1.groupby('case_name').apply(lambda x: \
#                       x['time:timestamp'].shift(periods=-1).fillna(t_date)))['time:timestamp'].values
#     print('Shift time at {0}'.format(str(datetime.now().time())))
    
#     sort_grup_1['time_diff'] = sort_grup_1['time:timestamp:2'] - sort_grup_1['time:timestamp']
#     print('Get difference time at {0}'.format(str(datetime.now().time())))
    
#     sort_grup_1['time_diff'] = sort_grup_1['time_diff'].apply(lambda x: 0 if x.days > 1000 else x.total_seconds())
#     print('Get difference time_2 at {0}'.format(str(datetime.now().time())))
    
#     sort_grup_1['transact'] = sort_grup_1[col_transact]+'-->'+sort_grup_1['concept:name:2']
#     print('Get transact at {0}'.format(str(datetime.now().time())))
    
#     return sort_grup_1[['transact', 'time_diff']].reset_index()



# def prepare_cases(df):
#     temp_df = pd.DataFrame(df)
#     print(temp_df.columns)
#     time_start = temp_df['time:timestamp'][0] - timedelta(seconds=1)
#     concept_name = 'Начало лога'
#     cs_name = temp_df.index.tolist()[0]
    
#     result = temp_df.append({'case_name': cs_name,
#                          'concept:name': concept_name,
#                          'time:timestamp': time_start}, ignore_index=True)
#     result = result.sort_values('time:timestamp')
#     return result


# def get_one_event_dict(one_event, case_name,data_types):

#     one_event_attri = list(one_event.keys())

#     one_event_dict = {}
#     for i in data_types:
#         if i in one_event_attri:
#             if type(one_event[i]) == list:
#                 for j in one_event[i]:
#                     one_event_dict[j['@key']] = j['@value']
#             else:
#                 one_event_dict[one_event[i]['@key']] = one_event[i]['@value']
#     one_event_dict['case_name'] = case_name
#     return one_event_dict

# def gain_one_trace_info(one_trace,data_types):
#     # for the attributer
#     one_trace_attri = list(one_trace.keys())
#     one_trace_attri_dict = {}

#     for i in data_types:
#         if i in one_trace_attri:
#             if type(one_trace[i]) == list:
#                 for j in one_trace[i]:
#                     one_trace_attri_dict[j['@key']] = j['@value']
#             else:
#                 one_trace_attri_dict[one_trace[i]['@key']] = one_trace[i]['@value']

#     # for event seq
#     one_trace_events = []
#     if type(one_trace['event']) == dict:
#         one_trace['event'] = [one_trace['event']]

#     for i in one_trace['event']:
#         inter_event = get_one_event_dict(i, one_trace_attri_dict['concept:name'],data_types)
#         one_trace_events.append(inter_event)

#     return one_trace_attri_dict,one_trace_events

# def gain_log_info_table(xml_string):
#     data_types = ['string', 'int', 'date', 'float', 'boolean', 'id']

#     log_is = xmltodict.parse(xml_string)
#     log_is = loads(dumps(log_is))

#     traces = log_is['log']['trace']

#     trace_attri = []
#     trace_event = []
#     j = 0
#     for i in traces:
#         inter = gain_one_trace_info(i,data_types)
#         trace_attri.append(inter[0])
#         trace_event = trace_event + inter[1]
#         j = j +1
# #         print(j)
#     return trace_attri,trace_event


# def line_color(transact, pend, not_pend, intersect):
#     color = ''
    
#     if transact in intersect:
#         color = 'yellow'
#     elif transact in pend:
#         color = 'green'
#     elif transact in not_pend:
#         color = 'red'
#     else:
#         color = 'black'
    
#     return color


# def line_width(transact, pend, not_pend):
#     width = 1
    
#     if transact in np.unique(pend + not_pend).tolist():
#         width = '8'
#     else:
#         width = '1'
    
#     return width


