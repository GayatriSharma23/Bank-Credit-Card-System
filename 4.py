import pandas as pd
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Graph
from pyecharts.globals import ThemeType

def build_family_relationship_graph(df, title='Family Relationship Knowledge Graph', repulsion=400, labelShow=True):
    """
    Build an interactive knowledge graph showing relationships between family_id and associated codes
    
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
        The rendered graph object with enhanced interactivity
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
            'category': category,
            # Add ID for referencing in JavaScript interactivity
            'id': entity
        })
    
    # Define links
    links = []
    for i in graph_df.index:
        links.append({
            'source': graph_df.loc[i, 'Node'],
            'target': graph_df.loc[i, 'Edge'],
            'value': graph_df.loc[i, 'Relation']
        })
    
    # Create a family to codes mapping for interactivity
    family_mapping = {}
    for idx, row in graph_df.iterrows():
        family = row['Node']
        code = row['Edge']
        if family not in family_mapping:
            family_mapping[family] = []
        family_mapping[family].append(code)
    
    # Create graph with enhanced interactivity - using compatible options
    g = (
        Graph(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="1000px", height="800px"))
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
            label_opts=opts.LabelOpts(is_show=labelShow),
            # Use the proper format for line styling
            linestyle_opts=opts.LineStyleOpts(
                width=2,
                curve=0.3
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            legend_opts=opts.LegendOpts(
                orient='vertical',
                pos_left='2%',
                pos_top='20%'
            ),
            # Add tooltip to show node details
            tooltip_opts=opts.TooltipOpts(
                trigger="item",
                formatter="{b}: {c}"
            )
        )
    )
    
    # Add custom JavaScript to implement the focus_node_adjacency functionality
    js_functions = """
    chart_CHART_ID.on('click', function(params) {
        if (params.dataType === 'node') {
            // Get all series data
            var seriesData = chart_CHART_ID.getOption().series[0];
            var nodes = seriesData.data;
            var links = seriesData.links;
            
            // Get the clicked node's name
            var nodeName = params.name;
            
            // Reset all nodes and links to default opacity
            for (var i = 0; i < nodes.length; i++) {
                nodes[i].itemStyle = {opacity: 0.3};
                nodes[i].label = {opacity: 0.3};
            }
            
            for (var i = 0; i < links.length; i++) {
                links[i].lineStyle = {opacity: 0.1, width: 1};
                links[i].label = {opacity: 0.1};
            }
            
            // If it's a family node, highlight this node and its connections
            if (nodeName.startsWith('Family_')) {
                // Highlight the clicked node
                for (var i = 0; i < nodes.length; i++) {
                    if (nodes[i].name === nodeName) {
                        nodes[i].itemStyle = {opacity: 1, borderWidth: 3, borderColor: '#000'};
                        nodes[i].label = {opacity: 1, fontWeight: 'bold'};
                        break;
                    }
                }
                
                // Find all connected nodes and highlight them
                var connectedNodes = [];
                for (var i = 0; i < links.length; i++) {
                    if (links[i].source === nodeName) {
                        connectedNodes.push(links[i].target);
                        links[i].lineStyle = {opacity: 1, width: 3};
                        links[i].label = {opacity: 1, fontWeight: 'bold'};
                    } else if (links[i].target === nodeName) {
                        connectedNodes.push(links[i].source);
                        links[i].lineStyle = {opacity: 1, width: 3};
                        links[i].label = {opacity: 1, fontWeight: 'bold'};
                    }
                }
                
                // Highlight the connected nodes
                for (var i = 0; i < nodes.length; i++) {
                    if (connectedNodes.includes(nodes[i].name)) {
                        nodes[i].itemStyle = {opacity: 1};
                        nodes[i].label = {opacity: 1, fontWeight: 'bold'};
                    }
                }
                
                // Update the chart
                chart_CHART_ID.setOption({series: [seriesData]});
            }
        }
    });
    """
    
    g.add_js_funcs(js_functions)
    
    return g

# Example usage:
# 1. Load your CSV data
# df = pd.read_csv('your_family_data.csv')
# 
# 2. Build and display the graph
# graph = build_family_relationship_graph(df)
# graph.render_notebook()  # For Jupyter notebook
# graph.render("family_graph.html")  # To save as HTML file

# Example sample data creation (for testing)
def create_sample_data():
    # Create sample data
    data = []
    for i in range(1, 6):  # 5 families
        owner = f"O_{i}"
        la = f"LA_{i}"
        payer = f"P_{i}" if i % 3 != 0 else None  # Some families don't have payers
        nominee1 = f"N1_{i}" if i % 2 == 0 else None  # Some families don't have nominees
        nominee2 = f"N2_{i}" if i % 4 == 0 else None
        
        data.append({
            'family_id': f"FAM{i}",
            'owner_code': owner,
            'la_code': la,
            'payer_code': payer,
            'nominee_1': nominee1,
            'nominee_2': nominee2
        })
    
    # Add some overlaps (same person has different roles in different families)
    data[3]['owner_code'] = data[0]['la_code']  # LA of family 1 is owner of family 4
    data[2]['nominee_1'] = data[1]['owner_code']  # Owner of family 2 is nominee of family 3
    
    return pd.DataFrame(data)

# Create sample data and generate graph
if __name__ == "__main__":
    df = create_sample_data()
    graph = build_family_relationship_graph(df, title="Interactive Family Relationships")
    graph.render("interactive_family_graph.html")
    print("Graph has been saved as 'interactive_family_graph.html'")

