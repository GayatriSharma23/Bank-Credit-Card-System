import pandas as pd
import numpy as np
from pyecharts.charts import Graph
from pyecharts import options as opts

def Build_graph(df, relation=False, repulsion=80, title='Knowledge Graph', labelShow=True):
    """
    Function to generate an interactive knowledge graph from family relationship data.
    """
    # Define category mappings for visualization
    color = {
        'FAMILY': '#6c5aab', 'OWNER': '#48D1CC', 'LA': '#e47596',
        'PAYER': '#1b8a69', 'NOMINEE': '#ff7f50'
    }

    category_mapping = {
        'FAMILY': 0, 'OWNER': 1, 'LA': 2, 'PAYER': 3, 'NOMINEE': 4
    }

    categories = [{'name': key, 'itemStyle': {'color': color[key]}} for key in category_mapping.keys()]

    # Identify all unique nodes
    unique_entities = set(df['Start_Entity']).union(set(df['End_Entity']))

    # Assign categories based on relation type
    entity_types = {}
    for _, row in df.iterrows():
        entity_types[row['Start_Entity']] = 'FAMILY'  # family_id is always FAMILY
        if 'OWNER' in row['Relation']:
            entity_types[row['End_Entity']] = 'OWNER'
        elif 'LA' in row['Relation']:
            entity_types[row['End_Entity']] = 'LA'
        elif 'PAYER' in row['Relation']:
            entity_types[row['End_Entity']] = 'PAYER'
        elif 'NOMINEE' in row['Relation']:
            entity_types[row['End_Entity']] = 'NOMINEE'

    # Define Nodes with Dynamic Sizing
    nodes = []
    for entity in unique_entities:
        category = category_mapping.get(entity_types.get(entity, 'FAMILY'), 0)
        size = max(10, np.log1p(df[(df['Start_Entity'] == entity) | (df['End_Entity'] == entity)].shape[0]) * 10)
        nodes.append({
            'name': entity,
            'symbolSize': size,  # Size based on number of connections
            'category': category
        })

    # Define Links
    links = [{'source': row['Start_Entity'], 'target': row['End_Entity'], 'value': row['Relation']} for _, row in df.iterrows()]

    # Build Graph
    graph = (
        Graph()
        .add(
            series_name="",
            nodes=nodes,
            links=links,
            categories=categories,
            repulsion=repulsion,
            layout="force",
            label_opts=opts.LabelOpts(is_show=labelShow),
            edge_label=opts.LabelOpts(is_show=True, position="middle", formatter="{c}")
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            legend_opts=opts.LegendOpts(orient='vertical', pos_left='2%', pos_top='40%', legend_icon='circle')
        )
        .render_notebook()
    )

    return graph
