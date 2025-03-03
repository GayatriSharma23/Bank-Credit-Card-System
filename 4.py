import random
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Graph
from pyecharts.globals import ThemeType, SymbolType
import webbrowser
import os

# Generate random data
def generate_random_data(num_records=20, max_nominees=3):
    data = []
    
    for i in range(num_records):
        family_id = f"Family_{random.randint(1, 10)}"
        record = {
            "owner_id": f"Owner_{random.randint(100, 999)}",
            "lacode": f"LA_{random.randint(100, 999)}",
            "payercode": f"Payer_{random.randint(100, 999)}",
            "family_id": family_id
        }
        
        # Add random number of nominees (1 to max_nominees)
        num_nominees = random.randint(1, max_nominees)
        for j in range(num_nominees):
            record[f"nominee{j+1}"] = f"Nominee_{random.randint(100, 999)}"
        
        data.append(record)
    
    return pd.DataFrame(data)

# Process data for graph visualization with labeled edges
def prepare_detailed_graph_data(df):
    nodes = []
    links = []
    node_ids = set()
    
    # Track categories for node types
    categories = [
        {"name": "Family", "itemStyle": {"color": "#FF6666"}},  # Red
        {"name": "Owner", "itemStyle": {"color": "#66B3FF"}},   # Blue
        {"name": "LA", "itemStyle": {"color": "#99FF99"}},      # Green
        {"name": "Payer", "itemStyle": {"color": "#FFCC99"}},   # Orange
        {"name": "Nominee", "itemStyle": {"color": "#CC99FF"}}  # Purple
    ]
    
    # Process each row in the dataframe
    for _, row in df.iterrows():
        family_id = row['family_id']
        
        # Add family node if not already added
        if family_id not in node_ids:
            nodes.append({
                "name": family_id,
                "symbolSize": 50,
                "category": 0,  # Category 0 is Family
                "value": family_id,
                "label": {"show": True, "position": "inside", "fontSize": 12, "fontWeight": "bold"},
                "emphasis": {"label": {"fontSize": 14}}
            })
            node_ids.add(family_id)
        
        # Process owner_id
        if 'owner_id' in row and pd.notna(row['owner_id']):
            owner_id = row['owner_id']
            if owner_id not in node_ids:
                nodes.append({
                    "name": owner_id,
                    "symbolSize": 35,
                    "category": 1,  # Category 1 is Owner
                    "value": owner_id,
                    "label": {"show": True, "fontSize": 10},
                    "emphasis": {"scale": True, "label": {"fontSize": 12}}
                })
                node_ids.add(owner_id)
            links.append({
                "source": owner_id, 
                "target": family_id,
                "value": "Owner",
                "lineStyle": {"width": 3, "curveness": 0.2},
                "emphasis": {"lineStyle": {"width": 6}}
            })
        
        # Process lacode
        if 'lacode' in row and pd.notna(row['lacode']):
            lacode = row['lacode']
            if lacode not in node_ids:
                nodes.append({
                    "name": lacode,
                    "symbolSize": 35,
                    "category": 2,  # Category 2 is LA
                    "value": lacode,
                    "label": {"show": True, "fontSize": 10},
                    "emphasis": {"scale": True, "label": {"fontSize": 12}}
                })
                node_ids.add(lacode)
            links.append({
                "source": lacode, 
                "target": family_id,
                "value": "LA Code",
                "lineStyle": {"width": 3, "curveness": 0.2},
                "emphasis": {"lineStyle": {"width": 6}}
            })
        
        # Process payercode
        if 'payercode' in row and pd.notna(row['payercode']):
            payercode = row['payercode']
            if payercode not in node_ids:
                nodes.append({
                    "name": payercode,
                    "symbolSize": 35,
                    "category": 3,  # Category 3 is Payer
                    "value": payercode,
                    "label": {"show": True, "fontSize": 10},
                    "emphasis": {"scale": True, "label": {"fontSize": 12}}
                })
                node_ids.add(payercode)
            links.append({
                "source": payercode, 
                "target": family_id,
                "value": "Payer",
                "lineStyle": {"width": 3, "curveness": 0.2},
                "emphasis": {"lineStyle": {"width": 6}}
            })
        
        # Process nominees
        for col in row.index:
            if col.startswith('nominee') and pd.notna(row[col]):
                nominee = row[col]
                if nominee not in node_ids:
                    nodes.append({
                        "name": nominee,
                        "symbolSize": 30,
                        "category": 4,  # Category 4 is Nominee
                        "value": nominee,
                        "label": {"show": True, "fontSize": 10},
                        "emphasis": {"scale": True, "label": {"fontSize": 12}}
                    })
                    node_ids.add(nominee)
                links.append({
                    "source": nominee, 
                    "target": family_id,
                    "value": f"Nominee ({col})",
                    "lineStyle": {"width": 3, "curveness": 0.2},
                    "emphasis": {"lineStyle": {"width": 6}}
                })
    
    return nodes, links, categories

# Create interactive graph with labeled edges
def create_detailed_relationship_graph(df, title="Customer Family Relationship Knowledge Graph"):
    nodes, links, categories = prepare_detailed_graph_data(df)
    
    # Create graph
    graph = (
        Graph(init_opts=opts.InitOpts(
            width="1200px", 
            height="800px", 
            theme=ThemeType.LIGHT,
            page_title=title
        ))
        .add(
            series_name="",
            nodes=nodes,
            links=links,
            categories=categories,
            layout="force",
            is_draggable=True,
            is_rotate_label=False,
            linestyle_opts=opts.LineStyleOpts(opacity=0.7, curve=0.3),
            label_opts=opts.LabelOpts(is_show=True, position="right", font_size=10),
            edge_label=opts.LabelOpts(
                is_show=True,
                position="middle",
                formatter="{c}",
                font_size=10,
                color="black",
                background_color="white",
                border_color="black",
                border_width=1
            ),
            gravity=0.3,
            repulsion=1000,
            edge_length=[150, 200],
            is_roam=True,
            is_focusnode=True
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=title,
                subtitle="Interactive visualization showing relationships between IDs"
            ),
            legend_opts=opts.LegendOpts(
                orient="horizontal", 
                pos_left="center", 
                pos_top="top",
                item_gap=25,
                item_width=25,
                item_height=15,
                textstyle_opts=opts.TextStyleOpts(font_size=12)
            ),
            toolbox_opts=opts.ToolboxOpts(
                is_show=True,
                orient="vertical",
                pos_left="right",
                feature=opts.ToolBoxFeatureOpts(
                    save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(title="Save as Image"),
                    data_view=opts.ToolBoxFeatureDataViewOpts(title="Data View", lang=["Data", "Close", "Refresh"]),
                    restore=opts.ToolBoxFeatureRestoreOpts(title="Restore"),
                    data_zoom=opts.ToolBoxFeatureDataZoomOpts(zoom_title="Zoom In", back_title="Zoom Out"),
                )
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="item", 
                formatter="{a} <br/>{b}: {c}"
            ),
        )
    )
    
    return graph

# Function to load data from Excel/CSV
def load_data(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        return pd.read_excel(file_path)

# Main function
def main():
    # Check if file exists, otherwise generate random data
    file_path = "customer_data.xlsx"
    
    if os.path.exists(file_path):
        print(f"Loading data from {file_path}")
        df = load_data(file_path)
    else:
        print(f"File {file_path} not found, generating random data")
        df = generate_random_data(num_records=20, max_nominees=3)
        # Optionally save the random data for reference
        df.to_excel("sample_data.xlsx", index=False)
        print("Random data saved to sample_data.xlsx")
    
    # Print a sample of the data
    print("\nSample of data:")
    print(df.head())
    
    # Create and save the graph
    graph = create_detailed_relationship_graph(df)
    
    # Save as HTML file
    html_file = "family_relationship_detailed_graph.html"
    graph.render(html_file)
    
    # Open in default browser
    webbrowser.open('file://' + os.path.realpath(html_file))
    
    print(f"\nDetailed graph saved as {html_file} and opened in browser")

if __name__ == "__main__":
    main()
