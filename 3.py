import pandas as pd

def transform_to_kg(df):
    """Transforms family structure DataFrame into knowledge graph format."""
    
    df_graph = pd.melt(
        df, id_vars=['family_id'],
        value_vars=['owner_code', 'la_code', 'payer_code', 'nominee_1_code', 'nominee_2_code', 'nominee_3_code'],
        var_name='Relation', value_name='Edge'
    )
    
    # Remove NaN values (i.e., families without certain roles)
    df_graph.dropna(subset=['Edge'], inplace=True)
    
    # Rename columns to fit knowledge graph structure
    df_graph.rename(columns={'family_id': 'Node'}, inplace=True)
    
    # Map relations
    df_graph['Relation'] = df_graph['Relation'].str.replace('_code', '').str.upper().replace({
        'OWNER': 'HAS_OWNER',
        'LA': 'HAS_LA',
        'PAYER': 'HAS_PAYER',
        'NOMINEE_1': 'HAS_NOMINEE',
        'NOMINEE_2': 'HAS_NOMINEE',
        'NOMINEE_3': 'HAS_NOMINEE'
    })

    # Add Start_Entity and End_Entity columns
    df_graph['Start_Entity'] = df_graph['Node']
    df_graph['End_Entity'] = df_graph['Edge']
    
    return df_graph[['Node', 'Relation', 'Edge', 'Start_Entity', 'End_Entity']]

# Example Usage
df = pd.DataFrame({
    'family_id': ['fam1', 'fam2', 'fam3'],
    'owner_code': ['O123', 'O456', None],
    'la_code': ['L456', None, 'L321'],
    'payer_code': ['P789', 'P222', 'P333'],
    'nominee_1_code': ['N001', 'N555', 'N666'],
    'nominee_2_code': ['N777', None, 'N888'],
    'nominee_3_code': [None, None, 'N999']
})

df_graph = transform_to_kg(df)
print(df_graph)
---------------------
import numpy as np
from pyecharts.charts import Graph
from pyecharts import options as opts

def Build_graph(df, relation=False, repulsion=40, title='Knowledge Graph', labelShow=False):
    """Generates a knowledge graph for family relationships."""
    
    entity_types = {'FAMILY': 0, 'OWNER': 1, 'LA': 2, 'PAYER': 3, 'NOMINEE': 4}
    color = {'FAMILY': '#6c5aab', 'OWNER': '#48D1CC', 'LA': '#e47596', 
             'PAYER': '#1b8a69', 'NOMINEE': '#ff7f50'}

    # Define categories for visualization
    categories = [{'name': key, 'itemStyle': {'normal': {'color': color[key]}}} for key in entity_types.keys()]

    # Extract unique entities
    unique_entities = set(df['Node']).union(set(df['Edge']))
    
    # Assign categories based on relation type
    entity_category = {entity: 'FAMILY' for entity in df['Node']}  # Families are main nodes
    
    for _, row in df.iterrows():
        if 'OWNER' in row['Relation']:
            entity_category[row['Edge']] = 'OWNER'
        elif 'LA' in row['Relation']:
            entity_category[row['Edge']] = 'LA'
        elif 'PAYER' in row['Relation']:
            entity_category[row['Edge']] = 'PAYER'
        elif 'NOMINEE' in row['Relation']:
            entity_category[row['Edge']] = 'NOMINEE'

    # Define Nodes
    nodes = []
    for entity in unique_entities:
        category = entity_types.get(entity_category.get(entity, 'FAMILY'), 0)
        size = max(10, np.log1p(df[(df['Node'] == entity) | (df['Edge'] == entity)].shape[0]) * 10)
        nodes.append({
            'name': entity,
            'symbolSize': size,  
            'category': category
        })

    # Define Links
    links = [{'source': row['Node'], 'target': row['Edge'], 'value': row['Relation']} for _, row in df.iterrows()]

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
