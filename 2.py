fig.update_layout(
    title_text="Parallel Categories Plot: Air Quality, Hospital Proximity, and Claims",
    title_font_size=20,
    showlegend=True,  # This will show the legend
    updatemenus=[
        {
            'buttons': [
                {
                    'label': 'Show Claims = 1',
                    'method': 'restyle',
                    'args': [{'line.color': [claims == 1 for claims in df['claims']]}, [0]]  # Filter by claims = 1 (green)
                },
                {
                    'label': 'Show Claims = 0',
                    'method': 'restyle',
                    'args': [{'line.color': [claims == 0 for claims in df['claims']]}, [0]]  # Filter by claims = 0 (red)
                },
                {
                    'label': 'Show All',
                    'method': 'restyle',
                    'args': [{'line.color': claims.tolist()}, [0]]  # Show all data
                }
            ],
            'direction': 'down',
            'pad': {'r': 10, 't': 10},
            'showactive': True,
            'x': 0.17,
            'xanchor': 'left',
            'y': 1.15,
            'yanchor': 'top'
        }
    ]
)
