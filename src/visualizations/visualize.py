import matplotlib.pyplot as plt

#image name
def visualize(name):
    
    vis_path = 'data/visualizations/'
    
    plt.savefig(vis_path + name)
    
    return None