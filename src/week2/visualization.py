import matplotlib.pyplot as plt

def plot_num_candidates(num_candidates_history):
    plt.figure(figsize=(7,4.5))
    plt.plot(num_candidates_history, linewidth=4)
    plt.xlabel('Search radius')
    plt.ylabel('# of documents searched')
    plt.rcParams.update({'font.size':16})
    plt.tight_layout()
    plt.show()

def plot_query_time(query_time_history):
    plt.figure(figsize=(7,4.5))
    plt.plot(query_time_history, linewidth=4)
    plt.xlabel('Search radius')
    plt.ylabel('Query time (seconds)')
    plt.rcParams.update({'font.size':16})
    plt.tight_layout()
    plt.show()

def plot_neighbors_distances(average_distance_from_query_history, max_distance_from_query_history, min_distance_from_query_history):
    plt.figure(figsize=(7,4.5))
    plt.plot(average_distance_from_query_history, linewidth=4, label='Average of 10 neighbors')
    plt.plot(max_distance_from_query_history, linewidth=4, label='Farthest of 10 neighbors')
    plt.plot(min_distance_from_query_history, linewidth=4, label='Closest of 10 neighbors')
    plt.xlabel('Search radius')
    plt.ylabel('Cosine distance of neighbors')
    plt.legend(loc='best', prop={'size':15})
    plt.rcParams.update({'font.size':16})
    plt.tight_layout()
    plt.show()
