import matplotlib.pyplot as plt
import numpy as np



def bar_plot_1(modality_comb: list[int], dice_metric: list[int],dataset: str,channels:list[str] ,combinations: int, name_note,indiv_mod:list[int],save: bool=False) -> None:
    """returns bar plot for the given modality_comb and dice metric"""
    try:
        assert len(modality_comb) == len(dice_metric)
    except:
        AssertionError("Length of modality_comb and dice_metric should be equal")

    fig, ax = plt.subplots()

    bar_width = 0.35
    index = np.arange(len(indiv_mod))
    bars = ax.bar(index, dice_metric, bar_width, label="Dice Score")
    ax.set_title(f"Bar Plot of Dice for {dataset} Different Modality Combinations")
    ax.set_xlabel("Modality")
    ax.set_ylabel("Dice")
    ax.set_xticks(index)
    ax.set_xticklabels([str(mod) for mod in indiv_mod], rotation=45, ha="right")
    ax.legend(title = str(channels) )
    # Add text annotations to the bars
    for bar, dice_score in zip(bars, dice_metric):
        height = bar.get_height()
        ax.annotate(f'{dice_score:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    fig.tight_layout()
    
    plt.tight_layout()
    plt.show()
    if save: 
        save_path =f'Exploratory_Analysis/'+'Bar_Plot_' + f'{dataset}_' + f'{name_note}_' f'{combinations}.png'
        fig.savefig(save_path,format='png',dpi= 300)
        
        print(f"\n ----------Bar plot saved at {save_path}----------\n")