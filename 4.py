import pandas as pd
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Graph
from pyecharts.globals import ThemeType

def build_family_relationship_graph(df, title='Family Relationship Knowledge Graph', repulsion=400, labelShow=True):
    """
    Build a knowledge graph showing relationships between family_id and associated codes
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing owner_code, la_code, payer_code, nominee columns and family_id
    title : str
        Title of the graph
    repulsion : int
        Repulsion factor for node spacing
    labelShow : bool
        Whether to show labels on nodes
    
    Returns:
    --------
    pyecharts.charts.Graph
        The rendered graph object
    """
    # Create a new dataframe for graph structure
    graph_data = []
    
    # Process each row to create connections
    for idx, row in df.iterrows():
        family_id = row['family_id']
        
        # Connect owner to family
        if pd.notna(row['owner_code']):
            graph_data.append({
                'Node': f"Family_{family_id}",
                'Edge': row['owner_code'],
                'Relation': 'Owner',
                'Entity_Type': 'Family_ID',
                'Code_Type': 'Owner'
            })
        
        # Connect LA to family
        if pd.notna(row['la_code']):
            graph_data.append({
                'Node': f"Family_{family_id}",
                'Edge': row['la_code'],
                'Relation': 'Life Assured',
                'Entity_Type': 'Family_ID',
                'Code_Type': 'LA'
            })
        
        # Connect payer to family
        if pd.notna(row['payer_code']):
            graph_data.append({
                'Node': f"Family_{family_id}",
                'Edge': row['payer_code'],
                'Relation': 'Payer',
                'Entity_Type': 'Family_ID',
                'Code_Type': 'Payer'
            })
        
        # Connect nominees to family
        for nominee_col in ['nominee_1', 'nominee_2', 'nominee_3']:
            if nominee_col in row and pd.notna(row[nominee_col]):
                graph_data.append({
                    'Node': f"Family_{family_id}",
                    'Edge': row[nominee_col],
                    'Relation': f'Nominee_{nominee_col[-1]}',
                    'Entity_Type': 'Family_ID',
                    'Code_Type': 'Nominee'
                })
    
    # Convert to DataFrame
    graph_df = pd.DataFrame(graph_data)
    
    # Define color scheme
    color = {
        'Family_ID': '#6c5aab',
        'Owner': '#48D1CC',
        'LA': '#e47596',
        'Payer': '#1b8a69',
        'Nominee': '#ff7f50'
    }
    
    # Define categories
    cate = {
        'Family_ID': 0,
        'Owner': 1,
        'LA': 2,
        'Payer': 3,
        'Nominee': 4
    }
    
    categories = [
        {'name': 'Family_ID', 'itemStyle': {'normal': {'color': color['Family_ID']}}},
        {'name': 'Owner', 'itemStyle': {'normal': {'color': color['Owner']}}},
        {'name': 'LA', 'itemStyle': {'normal': {'color': color['LA']}}},
        {'name': 'Payer', 'itemStyle': {'normal': {'color': color['Payer']}}},
        {'name': 'Nominee', 'itemStyle': {'normal': {'color': color['Nominee']}}}
    ]
    
    # Create entity type dictionary
    entity_type_dic = {}
    
    # Family IDs are of type Family_ID
    family_nodes = graph_df['Node'].unique()
    for node in family_nodes:
        entity_type_dic[node] = 'Family_ID'
    
    # Assign entity types to codes based on their relationship
    for idx, row in graph_df.drop_duplicates(['Edge']).iterrows():
        entity_type_dic[row['Edge']] = row['Code_Type']
    
    # Define nodes
    nodes = []
    all_entities = list(set(graph_df['Node']) | set(graph_df['Edge']))
    
    for entity in all_entities:
        # Calculate node size based on number of connections
        connections = graph_df.loc[(graph_df['Node'] == entity) | (graph_df['Edge'] == entity)].shape[0]
        symbol_size = max(15, np.log1p(connections) * 10)
        
        # Get entity type and corresponding category
        entity_type = entity_type_dic.get(entity, 'Unknown')
        category = cate.get(entity_type, 0)
        
        nodes.append({
            'name': entity,
            'symbolSize': symbol_size,
            'category': category
        })
    
    # Define links
    links = []
    for i in graph_df.index:
        links.append({
            'source': graph_df.loc[i, 'Node'],
            'target': graph_df.loc[i, 'Edge'],
            'value': graph_df.loc[i, 'Relation']
        })
    
    # Create and configure the graph
    g = (
        Graph(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
        .add(
            series_name="",
            nodes=nodes,
            links=links,
            categories=categories,
            repulsion=repulsion,
            edge_label=opts.LabelOpts(
                is_show=True,
                position="middle",
                formatter="{c}"
            ),
            layout="force",
            is_roam=True,
            label_opts=opts.LabelOpts(is_show=labelShow)
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            legend_opts=opts.LegendOpts(
                orient='vertical',
                pos_left='2%',
                pos_top='20%'
            )
        )
    )
    
    return g

# Example usage:
# 1. Load your CSV data
# df = pd.read_csv('your_family_data.csv')
# 
# 2. Build and display the graph
# graph = build_family_relationship_graph(df)
# graph.render_notebook()  # For Jupyter notebook
# graph.render("family_graph.html")  # To save as HTML file
