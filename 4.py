import random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from pyecharts import options as opts
from pyecharts.charts import Graph
from pyecharts.globals import ThemeType
import webbrowser
import os

# Generate random data
def generate_random_data(num_records=30, max_nominees=3):
    data = []
    
    for i in range(num_records):
        family_id = f"Family_{random.randint(1, 10)}"
        record = {
            "owner_id": f"Owner_{random.randint(1, 15)}",
            "lacode": f"LA_{random.randint(100, 999)}",
            "payercode": f"Payer_{random.randint(1, 20)}",
            "family_id": family_id
        }
        
        # Add random number of nominees (1 to max_nominees)
        num_nominees = random.randint(1, max_nominees)
        for j in range(num_nominees):
            record[f"nominee{j+1}"] = f"Nominee_{random.randint(1, 25)}"
        
        data.append(record)
    
    return pd.DataFrame(data)

# Create a static version for PowerPoint using matplotlib
def create_static_graph_for_ppt(df, output_file="family_relationship_graph_ppt.png"):
    # Create a directed graph
    G = nx.DiGraph()
    
    # Define node colors
    node_colors = {
        "Family": "#ff6666",   # Red
        "Owner": "#66b3ff",    # Blue
        "LA": "#99ff99",       # Green
        "Payer": "#ffcc99",    # Orange
        "Nominee": "#cc99ff"   # Purple
    }
    
    # Process each row in the dataframe
    for _, row in df.iterrows():
        family_id = row['family_id']
        
        # Add family node if not already added
        if not G.has_node(family_id):
            G.add_node(family_id, type="Family")
        
        # Process owner_id
        if 'owner_id' in row and pd.notna(row['owner_id']):
            owner_id = row['owner_id']
            if not G.has_node(owner_id):
                G.add_node(owner_id, type="Owner")
            G.add_edge(owner_id, family_id)
        
        # Process lacode
        if 'lacode' in row and pd.notna(row['lacode']):
            lacode = row['lacode']
            if not G.has_node(lacode):
                G.add_node(lacode, type="LA")
            G.add_edge(lacode, family_id)
        
        # Process payercode
        if 'payercode' in row and pd.notna(row['payercode']):
            payercode = row['payercode']
            if not G.has_node(payercode):
                G.add_node(payercode, type="Payer")
            G.add_edge(payercode, family_id)
        
        # Process nominees
        for col in row.index:
            if col.startswith('nominee') and pd.notna(row[col]):
                nominee = row[col]
                if not G.has_node(nominee):
                    G.add_node(nominee, type="Nominee")
                G.add_edge(nominee, family_id)
    
    # Prepare the visualization
    plt.figure(figsize=(14, 10))
    
    # Use a layout that works well for hierarchical data
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Create node colors list based on node type
    node_colors_list = [node_colors[G.nodes[node]['type']] for node in G.nodes()]
    
    # Create node sizes list based on node type
    node_sizes = [
        700 if G.nodes[node]['type'] == "Family" else
        500 if G.nodes[node]['type'] == "Owner" else
        500 if G.nodes[node]['type'] == "LA" else
        500 if G.nodes[node]['type'] == "Payer" else
        300  # Nominee
        for node in G.nodes()
    ]
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors_list, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5, arrows=True, arrowsize=15)
    nx.draw_networkx_labels(G, pos, font_size=9, font_family="sans-serif")
    
    # Add a legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=node_type)
        for node_type, color in node_colors.items()
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add a title
    plt.title("Customer Family Relationship Knowledge Graph", fontsize=16)
    
    # Remove axes
    plt.axis('off')
    
    # Save the image in high resolution
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Static graph for PowerPoint saved as {output_file}")
    
    return output_file

# Also create the interactive HTML version for live demo if needed
def create_interactive_html(df, output_file="family_relationship_graph.html"):
    # Process data for graph visualization
    nodes = []
    links = []
    node_ids = set()
    
    # Track categories for node types
    categories = [
        {"name": "Family"},
        {"name": "Owner"},
        {"name": "LA"},
        {"name": "Payer"},
        {"name": "Nominee"}
    ]
    
    # Process each row in the dataframe
    for _, row in df.iterrows():
        family_id = row['family_id']
        
        # Add family node if not already added
        if family_id not in node_ids:
            nodes.append({
                "name": family_id,
                "symbolSize": 30,
                "category": 0,  # Category 0 is Family
                "value": family_id
            })
            node_ids.add(family_id)
        
        # Process owner_id
        if 'owner_id' in row and pd.notna(row['owner_id']):
            owner_id = row['owner_id']
            if owner_id not in node_ids:
                nodes.append({
                    "name": owner_id,
                    "symbolSize": 20,
                    "category": 1,  # Category 1 is Owner
                    "value": owner_id
                })
                node_ids.add(owner_id)
            links.append({"source": owner_id, "target": family_id})
        
        # Process lacode
        if 'lacode' in row and pd.notna(row['lacode']):
            lacode = row['lacode']
            if lacode not in node_ids:
                nodes.append({
                    "name": lacode,
                    "symbolSize": 20,
                    "category": 2,  # Category 2 is LA
                    "value": lacode
                })
                node_ids.add(lacode)
            links.append({"source": lacode, "target": family_id})
        
        # Process payercode
        if 'payercode' in row and pd.notna(row['payercode']):
            payercode = row['payercode']
            if payercode not in node_ids:
                nodes.append({
                    "name": payercode,
                    "symbolSize": 20,
                    "category": 3,  # Category 3 is Payer
                    "value": payercode
                })
                node_ids.add(payercode)
            links.append({"source": payercode, "target": family_id})
        
        # Process nominees
        for col in row.index:
            if col.startswith('nominee') and pd.notna(row[col]):
                nominee = row[col]
                if nominee not in node_ids:
                    nodes.append({
                        "name": nominee,
                        "symbolSize": 15,
                        "category": 4,  # Category 4 is Nominee
                        "value": nominee
                    })
                    node_ids.add(nominee)
                links.append({"source": nominee, "target": family_id})
    
    # Create graph
    graph = (
        Graph(init_opts=opts.InitOpts(
            width="1200px", 
            height="800px", 
            theme=ThemeType.DARK,
            page_title="Customer Family Relationship Knowledge Graph"
        ))
        .add(
            series_name="",
            nodes=nodes,
            links=links,
            categories=categories,
            layout="force",
            is_draggable=True,
            linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.7, curve=0.3),
            label_opts=opts.LabelOpts(position="right", font_size=10),
            is_rotate_label=True,
            gravity=0.2,
            repulsion=50,
            edge_length=100,
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="Customer Family Relationship Knowledge Graph",
                subtitle="Interactive visualization of customer relationships"
            ),
            legend_opts=opts.LegendOpts(
                orient="vertical", 
                pos_left="2%", 
                pos_top="20%"
            ),
            toolbox_opts=opts.ToolboxOpts(
                feature=opts.ToolBoxFeatureOpts(
                    save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(title="Save as Image"),
                    data_view=opts.ToolBoxFeatureDataViewOpts(title="View Data", lang=["Data View", "Close", "Refresh"]),
                    restore=opts.ToolBoxFeatureRestoreOpts(title="Restore"),
                    data_zoom=opts.ToolBoxFeatureDataZoomOpts(zoom_title="Zoom In", back_title="Zoom Out"),
                )
            ),
        )
    )
    
    # Save as HTML file
    graph.render(output_file)
    print(f"Interactive HTML graph saved as {output_file}")
    
    return output_file

# Main function
def main():
    # Generate random data
    df = generate_random_data(num_records=30, max_nominees=3)
    
    # Print a sample of the data
    print("Sample of generated data:")
    print(df.head())
    
    # Create both versions
    static_file = create_static_graph_for_ppt(df)
    html_file = create_interactive_html(df)
    
    # Create multiple static images with different layouts for PowerPoint slides
    # Create a few variations with focused subgraphs
    families = df['family_id'].unique()
    if len(families) >= 3:
        for i in range(min(3, len(families))):
            family_subset = df[df['family_id'] == families[i]]
            create_static_graph_for_ppt(family_subset, f"family_{families[i]}_detail.png")
    
    print("\nCreated files for your presentation:")
    print(f"1. {static_file} - Static image to embed in PowerPoint")
    print(f"2. {html_file} - Interactive HTML for live demo")
    print("3. Detail views of individual families for slide transitions")
    
    # Open HTML in browser for viewing
    webbrowser.open('file://' + os.path.realpath(html_file))

if __name__ == "__main__":
    main()
    
