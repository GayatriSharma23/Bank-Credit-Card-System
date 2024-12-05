import plotly.express as px
import pandas as pd

def plot_dynamic_bar_chart(df):
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    if len(numeric_cols) == 0 or len(categorical_cols) == 0:
        raise ValueError("DataFrame must have at least one numeric and one categorical column.")

    # Select the first numeric and categorical column
    x_col = numeric_cols[0]
    y_col = categorical_cols[0]

    # Ensure categorical column is treated as category
    df[y_col] = df[y_col].astype('category')

    # Create the bar plot
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        orientation='h',  # Horizontal bar plot
        text=x_col  # Display the count as text on the bar
    )

    # Update layout for better visualization
    fig.update_layout(
        title=f"Bar Plot of {x_col} by {y_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        xaxis=dict(type='linear'),  # Ensure x-axis is numeric
        yaxis=dict(categoryorder='total ascending')  # Sort bars by count
    )

    return fig

# Example DataFrame
data = {
    "itemscount": [343, 2345, 10, 1456],
    "custstate": ["punjab", "up", "gujarat", "mp"]
}
df = pd.DataFrame(data)

# Generate the bar plot
fig = plot_dynamic_bar_chart(df)
fig.show()
