df_graph = pd.melt(df, id_vars=['family_id'], value_vars=['owner_code', 'la_code', 'payer_code', 'nominee_1_code', 'nominee_2_code', 'nominee_3_code'],
                   var_name='Relation', value_name='End_Entity')

df_graph['Relation'] = df_graph['Relation'].str.replace('_code', '').str.upper().replace({
    'OWNER': 'HAS_OWNER',
    'LA': 'HAS_LA',
    'PAYER': 'HAS_PAYER',
    'NOMINEE_1': 'HAS_NOMINEE',
    'NOMINEE_2': 'HAS_NOMINEE',
    'NOMINEE_3': 'HAS_NOMINEE'
})

df_graph.rename(columns={'family_id': 'Start_Entity'}, inplace=True)
 -----------------------------------------------------
import pandas as pd
import numpy as np
from pyecharts.charts import Graph
from pyecharts import options as opts

def Build_graph(df, relation=False, repulsion=40, title='Knowledge Graph', labelShow=False):
    # Mapping entity types (family, owner, etc.) for visualization
    entity_type_dic = dict(df.drop_duplicates(['End_Entity']).set_index(['End_Entity'])['Start_Entity'])

    color = {
        'FAMILY': '#6c5aab', 'OWNER': '#48D1CC', 'LA': '#e47596', 
        'PAYER': '#1b8a69', 'NOMINEE': '#ff7f50'
    }

    cate = {'FAMILY': 0, 'OWNER': 1, 'LA': 2, 'PAYER': 3, 'NOMINEE': 4}
    
    categories = [{'name': key, 'itemStyle': {'normal': {'color': color[key]}}} for key in cate.keys()]
    
    # Define Nodes
    nodes = []
    unique_entities = set(df['Start_Entity']).union(set(df['End_Entity']))
    
    for entity in unique_entities:
        category = cate.get(entity_type_dic.get(entity, 'FAMILY'), 0)
        nodes.append({
            'name': entity,
            'symbolSize': max(10, np.log1p(df.loc[(df['Start_Entity']==entity) | (df['End_Entity']==entity)].shape[0]) * 10 // 1),
            'category': category
        })
    
    # Define Links
    links = []
    for i in df.index:
        links.append({'source': df.loc[i, 'Start_Entity'], 'target': df.loc[i, 'End_Entity'], 'value': df.loc[i, 'Relation']})
    
    # Build Graph
    g = (
        Graph()
        .add('', nodes, links, categories, repulsion=repulsion, label_opts=opts.LabelOpts(is_show=labelShow))
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            legend_opts=opts.LegendOpts(orient='vertical', pos_left='2%', pos_top='40%', legend_icon='circle')
        )
        .render_notebook()
    )
    
    return g
