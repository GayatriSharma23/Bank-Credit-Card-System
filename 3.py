import networkx as nx
from pyvis.network import Network
import pandas as pd

# Sample Data
data = [
    {"Policy Number": "P1", "Owner": "A", "Payer": "B", "Nominee1": "C", "Nominee2": "D", "Nominee3": None, "LA": "E", "Family ID": "F1"},
    {"Policy Number": "P2", "Owner": "A", "Payer": "F", "Nominee1": "G", "Nominee2": None, "Nominee3": None, "LA": "H", "Family ID": "F1"},
    {"Policy Number": "P3", "Owner": "I", "Payer": "J", "Nominee1": "K", "Nominee2": "L", "Nominee3": "M", "LA": "N", "Family ID": "F2"},
]

df = pd.DataFrame(data)

# Create PyVis Network
net = Network(height="700px", width="100%", notebook=True, directed=False)
net.toggle_hide_edges_on_drag(True)  # Hide edges when dragging for clarity
net.toggle_hide_nodes_on_drag(True)  # Hide nodes when dragging

# Define Colors
colors = {
    "Policy": "blue",
    "Owner": "red",
    "Payer": "green",
    "Nominee": "orange",
    "LA": "purple",
    "Family": "black",
}

# Add Family ID Nodes
for family_id in df["Family ID"].unique():
    net.add_node(family_id, label=f"Family {family_id}", color=colors["Family"], shape="box", size=20)

# Add Nodes and Edges with Filtering Feature
for _, row in df.iterrows():
    policy = row["Policy Number"]
    family_id = row["Family ID"]

    net.add_node(policy, label=policy, color=colors["Policy"], size=15)
    net.add_edge(family_id, policy, label="Has Policy")

    for role in ["Owner", "Payer", "Nominee1", "Nominee2", "Nominee3", "LA"]:
        if row[role]:  # Ignore None values
            entity_type = "Nominee" if "Nominee" in role else role
            net.add_node(row[role], label=row[role], color=colors[entity_type], size=10)
            net.add_edge(policy, row[role], label=role)

# Enable hierarchical structure for better visualization
net.barnes_hut()
net.show_buttons(filter_=['physics'])  # Enable physics settings control

# Generate and show the graph
net.show("family_graph.html")

