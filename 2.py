import numpy as np
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Graph
from pyecharts.render import make_snapshot
from IPython.display import display, HTML

# Function to build the knowledge graph
def Build_graph(df, relation=False, repulsion=80, title='Knowledge Graph', labelShow=True, output_file='knowledge_graph.html'):
    """
    Build and render a knowledge graph from a dataframe of relationships.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Node, Edge, Start_Entity, End_entity columns.
    relation : bool, default=False
        Whether to show relation values on edges.
    repulsion : int, default=80
        Node repulsion strength for layout.
    title : str, default='Knowledge Graph'
        Title of the graph.
    labelShow : bool, default=True
        Whether to show labels on nodes.
    output_file : str, default='knowledge_graph.html'
        Output HTML file path.

    Returns:
    --------
    None
    """

    # Define color scheme for entity types
    color_map = {
        'RESIDES IN': '#6c5aab', 'WORKS AS': '#48D1CC', 'PRODUCT_CATEGORY': '#e47596',
        'CHANNEL': '#1b8a69', 'LA': '#ff7f50', 'EDUCATION': '#5084ff',
        'PRODUCT NAME': '#25162b', 'STAT_CLAIM': '#ab945a', 'AGE': '#77DD77'
    }

    # Category mapping
    cate = {key: i for i, key in enumerate(color_map.keys())}

    # Define categories for graph styling
    categories = [{"name": key, "itemStyle": {"color": color_map[key]}} for key in color_map.keys()]

    # Map entity types
    entity_type_dic = df.drop_duplicates(['Node']).set_index(['Node'])['Start_Entity'].to_dict()
    entity_type_dic.update(df.drop_duplicates(['Edge']).set_index(['Edge'])['End_entity'].to_dict())

    # Create nodes with calculated sizes
    nodes = []
    for entity in set(df['Node']).union(set(df['Edge'])):
        if entity in entity_type_dic and entity_type_dic[entity] in cate:
            size = max(10, np.log1p(df[(df['Node'] == entity) | (df['Edge'] == entity)].shape[0]) * 10 // 1)
            nodes.append({
                "name": entity,
                "symbolSize": size,
                "category": cate[entity_type_dic[entity]],
                "itemStyle": {"color": color_map.get(entity_type_dic[entity], "#cccccc")}
            })

    # Create links (edges) with optional relation labels
    links = []
    for _, row in df.iterrows():
        link_data = {"source": row["Node"], "target": row["Edge"]}
        if relation and "pred" in df.columns:
            link_data["value"] = row["pred"]
        links.append(link_data)

    # Define graph styles
    graph = (
        Graph()
        .add(
            "",
            nodes,
            links,
            categories=categories,
            repulsion=repulsion,
            is_roam=True,
            is_focusnode=True,
            label_opts=opts.LabelOpts(is_show=labelShow, position="right", font_size=12),
            linestyle_opts=opts.LineStyleOpts(width=2, curve=0.3, opacity=0.8),
            edge_symbol=['circle', 'arrow'],
            edge_symbol_size=[5, 12],
            edge_label=opts.LabelOpts(is_show=relation, formatter="{c}", font_size=10, color="#ff0000"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title, title_textstyle_opts=opts.TextStyleOpts(font_size=20)),
            tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{b}"),
            legend_opts=opts.LegendOpts(pos_left="5%", pos_top="15%", item_gap=20, item_width=30, item_height=16),
        )
    )

    # Render graph to HTML
    output_path = graph.render(output_file)
    print(f"Graph saved to: {output_path}")

    # Render graph in Jupyter Notebook
    display(HTML(output_file))  # This will display the saved HTML in Jupyter Notebook


Build_graph(df, relation=True, output_file="knowledge_graph.html")
