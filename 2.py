def Build_graph(df, relation=False, repulsion=40, title='Knowledge Graph', labelShow=False, output_file='knowledge_graph.html'):
    """
    Build and render a knowledge graph from a dataframe of relationships.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Node, Edge, Start_Entity, End_entity columns
    relation : bool, default=False
        Whether to show relation values on edges
    repulsion : int, default=40
        Node repulsion strength for layout
    title : str, default='Knowledge Graph'
        Title of the graph
    labelShow : bool, default=False
        Whether to show labels on nodes
    output_file : str, default='knowledge_graph.html'
        Output HTML file path
    
    Returns:
    --------
    str
        Path to the rendered HTML file
    """
    import numpy as np
    from pyecharts import options as opts
    from pyecharts.charts import Graph
    
    # Create dictionary mapping entities to their types
    entity_type_dic = {}
    
    # Handle potential KeyError by checking if columns exist
    if 'Node' in df.columns and 'Start_Entity' in df.columns:
        node_types = df.drop_duplicates(['Node']).set_index(['Node'])['Start_Entity'].to_dict()
        entity_type_dic.update(node_types)
    
    if 'Edge' in df.columns and 'End_entity' in df.columns:
        edge_types = df.drop_duplicates(['Edge']).set_index(['Edge'])['End_entity'].to_dict()
        entity_type_dic.update(edge_types)
    
    # Define colors for different entity types
    color = {
        'RESIDES IN': '#6c5aab',
        'WORKS AS': '#48D1CC',
        'PRODUCT_CATEGORY': '#e47596',
        'CHANNEL': '#1b8a69',
        'LA': '#ff7f50',
        'EDUCATION': '#5084ff',
        'PRODUCT NAME': '#25162b',
        'STAT_CLAIM': '#ab945a',
        'AGE': '#77DD77'
    }
    
    # Map entity types to category indices
    cate = {
        'RESIDES IN': 0,
        'WORKS AS': 1,
        'PRODUCT_CATEGORY': 2,
        'CHANNEL': 3,
        'LA': 4,
        'EDUCATION': 5,
        'PRODUCT NAME': 6,
        'STAT_CLAIM': 7,
        'AGE': 8
    }
    
    # Define categories with their styles
    categories = [
        {'name': 'RESIDES IN', 'itemStyle': {'normal': {'color': color['RESIDES IN']}}},
        {'name': 'WORKS AS', 'itemStyle': {'normal': {'color': color['WORKS AS']}}},
        {'name': 'AGE', 'itemStyle': {'normal': {'color': color['AGE']}}},
        {'name': 'PRODUCT_CATEGORY', 'itemStyle': {'normal': {'color': color['PRODUCT_CATEGORY']}}},
        {'name': 'CHANNEL', 'itemStyle': {'normal': {'color': color['CHANNEL']}}},
        {'name': 'LA', 'itemStyle': {'normal': {'color': color['LA']}}},
        {'name': 'EDUCATION', 'itemStyle': {'normal': {'color': color['EDUCATION']}}},
        {'name': 'PRODUCT NAME', 'itemStyle': {'normal': {'color': color['PRODUCT NAME']}}},
        {'name': 'STAT_CLAIM', 'itemStyle': {'normal': {'color': color['STAT_CLAIM']}}}
    ]
    
    # Debug info
    print(f"Total unique nodes and edges: {len(set(df['Node']) | set(df['Edge']))}")
    
    # Define nodes with error handling
    nodes = []
    for entity in list(set(df['Node']) | set(df['Edge'])):
        # Check if entity exists in entity_type_dic
        if entity in entity_type_dic and entity_type_dic[entity] in cate:
            # Calculate node size based on frequency
            size = max(10, np.log1p(df.loc[(df['Node'] == entity) | (df['Edge'] == entity)].shape[0]) * 10 // 1)
            
            nodes.append({
                'name': entity, 
                'symbolSize': size,
                'category': cate[entity_type_dic[entity]]
            })
        else:
            print(f"Warning: Entity {entity} not found in mappings, or its type is not in category map.")
    
    # Define links with error handling
    links = []
    for i in df.index:
        node = df.loc[i, 'Node']
        edge = df.loc[i, 'Edge']
        
        # Check if both node and edge exist in nodes
        node_exists = any(n['name'] == node for n in nodes)
        edge_exists = any(n['name'] == edge for n in nodes)
        
        if node_exists and edge_exists:
            if not relation:
                links.append({
                    'source': node, 
                    'target': edge
                })
            else:
                # Check if 'pred' column exists
                if 'pred' in df.columns:
                    links.append({
                        'source': node, 
                        'target': edge, 
                        'value': df.loc[i, 'pred']
                    })
                else:
                    links.append({
                        'source': node, 
                        'target': edge
                    })
        else:
            print(f"Warning: Link between {node} and {edge} skipped due to missing node.")
    
    print(f"Created {len(nodes)} nodes and {len(links)} links.")
    
    # Additional styling options for a more appealing graph
    label_opts = opts.LabelOpts(
        is_show=labelShow,
        position="right",
        formatter="{b}",
        font_size=12,
        font_style="normal",
        font_weight="normal",
    )
    
    # Edge styling
    linestyle_opts = opts.LineStyleOpts(
        width=1,
        curve=0.3,
        opacity=0.7,
    )
    
    # Create and configure the graph
    g = (
        Graph()
        .add(
            series_name="", 
            nodes=nodes, 
            links=links, 
            categories=categories, 
            repulsion=repulsion,
            is_roam=True,  # Allow zooming and panning
            is_focusnode=True,  # Highlight nodes on hover
            label_opts=label_opts,
            linestyle_opts=linestyle_opts,
            is_rotate_label=True,  # Allow label rotation for better readability
            layout="force",  # Use force-directed layout
            edge_symbol=['circle', 'arrow'],  # Add arrows to show direction
            edge_symbol_size=[4, 10],  # Size of the edge symbols
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=title,
                subtitle="Interactive Knowledge Graph",
                pos_left="center",
                title_textstyle_opts=opts.TextStyleOpts(font_size=20),
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="item", 
                formatter="{a} <br/>{b} --> {c}"
            ),
            legend_opts=opts.LegendOpts(
                orient='vertical', 
                pos_left='2%', 
                pos_top='20%',
                legend_icon='circle',
                is_show=True,
                item_gap=20,
                item_width=25,
                item_height=14,
                textstyle_opts=opts.TextStyleOpts(font_size=12),
            )
        )
    )
    
    # Render to HTML file
    output_path = g.render(output_file)
    print(f"Graph rendered to {output_path}")
    
    return output_path
