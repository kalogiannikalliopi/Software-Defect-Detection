import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class plot_results:
    
    def __init__(self, dict_normalizer, normalizers_name, dict_models, dataset):
        self.dict_normalizer = dict_normalizer
        self.normalizers_name = normalizers_name
        self.dict_models = dict_models
        self.dataset = dataset
        
    def addlabels(self, x,y):
        for i in range(len(x)):
            plt.text(x[i], y[i]+0.05*y[i], round(y[i],3), ha = 'center')
        
    def bar(self):
        
        barWidth = 0.3
        fig = plt.subplots(figsize =(18, 4))

        # Set the y-axis maximum value
        plt.ylim(0, max(max(list(self.dict_normalizer.values())[0]), max(list(self.dict_normalizer.values())[1]), max(list(self.dict_normalizer.values())[2])) * 1.2)

        # Set position of bar on X axis 
        br1 = np.arange(len(list(self.dict_normalizer.values())[0]))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]

        # Make the plot
        plt.bar(br1, list(self.dict_normalizer.values())[0], color ='steelblue', width = barWidth,
                edgecolor ='k', linewidth=1.5, label =list(self.dict_normalizer.keys())[0])
        self.addlabels(br1,list(self.dict_normalizer.values())[0])

        plt.bar(br2, list(self.dict_normalizer.values())[1], color ='chocolate', width =barWidth,
                edgecolor ='k', linewidth=1.5, label =list(self.dict_normalizer.keys())[1])
        self.addlabels(br2,list(self.dict_normalizer.values())[1])

        plt.bar(br3, list(self.dict_normalizer.values())[2], color ='forestgreen', width = barWidth, 
                edgecolor ='k', linewidth=1.5, label =list(self.dict_normalizer.keys())[2])
        self.addlabels(br3,list(self.dict_normalizer.values())[2])

        # Adding Xticks 
        plt.xlabel('Classifier', fontweight ='bold', fontsize = 15) 
        plt.ylabel(self.normalizers_name, fontweight ='bold', fontsize = 15) 
        plt.xticks([r + barWidth for r in range(len(list(self.dict_models.keys())))], 
                list(self.dict_models.keys()))

        plt.legend(loc='upper left', bbox_to_anchor=(0.5, 1.2), fancybox=True, shadow=True, ncol=len(self.dict_normalizer.keys()))

        # Create folder if it doesn't exist
        folder = Path("results") / self.dataset
        folder.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()

        # Save plot
        file_path = folder / f"{self.normalizers_name}.png"
        plt.savefig(file_path)
        plt.show()
        plt.close()    