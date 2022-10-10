# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 14:42:08 2021

@author: Mark Marner-Hausen
"""


# This is a function to create simple line plos with some measure of adjustability
def lineplot_price(data, vertical, horizontal, wide, long, name_figure, description, rounds, text_size, x_axis, y_axis, identifier,
                   label_xaxis, label_yaxis, color):
    
    
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import seaborn as sns
    
    # Specifies how many plots are displayed besides and above of each other, 
    # As well as the size of the overall plot 
    # NOTE: The plot is based on an indentifier called 'rounds'
    figX, axes = plt.subplots(vertical , horizontal, figsize=(wide,long))
    h = 0
    for i in range(0,vertical):
        for j in range(0, horizontal):
            # 'rounds' specifies the fraction of data used in each subplot
                tmp_hlep = rounds[h]
                tmp = data[data.rounds == tmp_hlep]
                if vertical == 1:
                    # This is the actual subplot
                    figX = sns.lineplot(x=x_axis, y=y_axis, data=tmp, hue=identifier,
                                        palette=color, ax = axes[j], legend=False)
                    
                    
                    axes[j].set_title(f'Round {tmp_hlep}', fontsize = text_size)
                    
                    # Not every plot displays a label for x and y axis
                    axes[j].set_xlabel(None)
                    axes[j].set_ylabel(None)
                    h = h+1
                    
                else: 
                    # This is the actual subplot
                    figX = sns.lineplot(x=x_axis, y=y_axis, data=tmp, hue=identifier,
                                        palette=color, ax = axes[i, j], legend=False)
                    axes[i, j].set_title(f'Round {tmp_hlep}', fontsize = text_size)
                    
                    # Not every plot displays a label for x and y axis
                    axes[i, j].set_xlabel(None)
                    axes[i, j].set_ylabel(None)
                    h = h + 1
                   
      
    if vertical == 1:    
    
        # Legend is only shown in top left subplot         
        axes[0].legend(loc='lower left',  title = None, fontsize= text_size, labels = description)
        
        # Specifies were x axis should be labeled 
        for i in range(0, horizontal):
            axes[i].set_xlabel(label_xaxis, fontsize = text_size)
       
        # Specifies were y acis should be labeled 
        for i in range(0, 1):
             axes[0].set_ylabel(label_yaxis, fontsize = text_size)
             
    else:
        
         # Legend is only shown in top left subplot         
        axes[0,0].legend(loc='lower left',  title = None, fontsize= text_size, labels = description)
        
        # Specifies were x axis should be labeled 
        for i in range(0, horizontal):
            axes[vertical-1, i].set_xlabel(label_xaxis, fontsize = text_size)
       
        # Specifies were y acis should be labeled 
        for i in range(0, vertical):
             axes[i, 0].set_ylabel(label_yaxis, fontsize = text_size)
         
    # Should ensure that figure is displyed within Python     
    figX = figX.get_figure()
    plt.tight_layout()
    plt.show()
    figX.savefig(f'{name_figure}.pdf')