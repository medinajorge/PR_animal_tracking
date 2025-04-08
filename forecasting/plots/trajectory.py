import numpy as np
from phdu import savefig
from tidypath import storage
import plotly.graph_objects as go
from phdu.plots.plotly_utils import plotly_default_colors
from tqdm.notebook import tqdm

@savefig("figname", ext="html")
def trajectories_visualization(height=1500, trajectories=None, species=None, figname='all', filter_func=lambda x,y,z: True):
    trajectories = storage.load_lzma('utils/data/trajectories_default.lzma') if trajectories is None else trajectories
    species = storage.load_lzma('utils/data/labels_default.lzma')['COMMON_NAME'] if species is None else species
    species_counts = species.value_counts()
    species_names = species_counts.index
    colors = plotly_default_colors()[:len(species_names)]
    species_to_color = {s:c for s,c in zip(species_names, colors)}
    fig = go.Figure(go.Scattergeo())
    fig.update_geos(
        projection_type="orthographic",
        resolution=50,
        showcoastlines=True, coastlinecolor="RebeccaPurple",
        showland=True, landcolor="LightGreen",
        showocean=True, oceancolor="LightBlue"
    )
    fig.update_layout(height=height, margin={"r":0,"t":0,"l":0,"b":0})

    species_drawn = set()
    trajs_drawn = 0
    for t, s in zip(trajectories, tqdm(species)):
        if filter_func(t, s, species_counts):
            plot_idxs = np.linspace(0, t.shape[1]-1, 20, dtype=np.int32) if t.shape[1] > 20 else [*range(t.shape[1])]
            fig.add_trace(go.Scattergeo(
                    lat = t[0][plot_idxs],
                    lon = t[1][plot_idxs],
                    mode = 'lines',
                    line = dict(width = 1, color = species_to_color[s]),
                    opacity = 1,
                    name = s,
                    showlegend = s not in species_drawn
                    )
            )
            species_drawn.add(s)
            trajs_drawn += 1
        else:
            continue
    fig.update_layout(legend_title_text='Species', showlegend=True, legend_font_size=10)
    return fig
