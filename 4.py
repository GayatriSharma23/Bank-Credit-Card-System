import random
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Graph
from pyecharts.globals import ThemeType
import webbrowser
import os
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot

# Generate random data
def generate_random_data(num_records=50, max_nominees=3):
    data = []
    
    for i in range(num_records):
        family_id = f"Family_{random.randint(1, 15)}"
        record = {
            "owner_id": f"Owner_{random.randint(1, 30)}",
            "lacode": f"LA_{random.randint(100, 999)}",
            "payercode": f"Payer_{random.randint(1, 40)}",
            "family_id": family_id
        }
        
        # Add random number of nominees (1 to max_nominees)
        num_nominees = random.randint(1, max_nominees)
        for j in range(num_nominees):
            record[f"nominee{j+1}"] = f"Nominee_{random.randint(1, 50)}"
        
        data.append(record)
    
    return pd.DataFrame(data)

# Process data for graph visualization
def prepare_graph_data(df):
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
    
    return nodes, links, categories

# Create interactive graph
def create_relationship_graph(df, title="Customer Family Relationship Knowledge Graph"):
    nodes, links, categories = prepare_graph_data(df)
    
    # Create graph
    graph = (
        Graph(init_opts=opts.InitOpts(
            width="1200px", 
            height="800px", 
            theme=ThemeType.DARK,
            page_title=title
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
                title=title,
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
    
    return graph

# Main function
def main():
    # Generate random data
    df = generate_random_data(num_records=50, max_nominees=3)
    
    # Print a sample of the data
    print("Sample of generated data:")
    print(df.head())
    
    # Create and save the graph
    graph = create_relationship_graph(df)
    
    # Save as HTML file
    html_file = "family_relationship_graph.html"
    graph.render(html_file)
    
    # Open in default browser
    webbrowser.open('file://' + os.path.realpath(html_file))
    
    # Optionally, save as image
    make_snapshot(snapshot, graph.render(), "family_relationship_graph.png")
    
    print(f"\nGraph saved as {html_file} and opened in browser")
    print("Image snapshot saved as family_relationship_graph.png")

if __name__ == "__main__":
    main()
