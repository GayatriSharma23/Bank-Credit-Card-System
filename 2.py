def Build_graph(df, relation=False, repulsion=40, title='Knowledge Graph', labelShow=False, output_file='knowledge_graph.html'):
    # Create dictionary mapping entities to their types
    entity_type_dic = dict(df.drop_duplicates(['Node']).set_index(['Node'])['Start_Entity'])
    entity_type_dic.update(df.drop_duplicates(['Edge']).set_index(['Edge'])['End_entity'])
    
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
    
    # Define nodes
    nodes = []
    for entity in list(set(df['Node']) | set(df['Edge'])):
        nodes.append({
            'name': entity, 
            'symbolSize': max(10, np.log1p(df.loc[(df['Node'] == entity) | (df['Edge'] == entity)].shape[0]) * 10 // 1),
            'category': cate[entity_type_dic[entity]]
        })
        
    # Define links
    links = []
    for i in df.index:
        if not relation:
            links.append({
                'source': df.loc[i, 'Node'], 
                'target': df.loc[i, 'Edge']
            })
        else:
            links.append({
                'source': df.loc[i, 'Node'], 
                'target': df.loc[i, 'Edge'], 
                'value': df.loc[i, 'pred']
            })
    
    # Create and configure the graph
    g = (
        Graph()
        .add(
            '', 
            nodes, 
            links, 
            categories, 
            repulsion=repulsion,
            label_opts=opts.LabelOpts(is_show=labelShow)
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            legend_opts=opts.LegendOpts(
                orient='vertical', 
                pos_left='2%', 
                pos_top='40%',
                legend_icon='circle'
            )
        )
        .render(output_file)  # Changed from render_notebook() to render() with a file path
    )
    
    return output_file
