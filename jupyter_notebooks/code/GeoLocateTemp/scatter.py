import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

df = pd.read_csv('metadata.csv')
df = df.dropna(subset=['Actual_Latitude', 'Actual_Longitude'])
df = df[pd.to_numeric(df['Actual_Latitude'], errors='coerce').notnull()]
df = df[pd.to_numeric(df['Actual_Longitude'], errors='coerce').notnull()]
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)

ax.scatter(
    x=df['Actual_Longitude'],
    y=df['Actual_Latitude'],
    color='red',
    marker='o',
    s=5,
    alpha=0.5,
    transform=ccrs.PlateCarree(),
    label='Data Points'
)
# Add gridlines and labels
ax.gridlines(draw_labels=True, linestyle='--')

# Add title and legend
plt.title('Geographical Points Visualization')
plt.legend()
plt.savefig(
    'scatterplot.pdf',  # File name
    dpi=600,          # Image resolution (300-600 for publications)
    bbox_inches='tight',  # Prevent cropped borders
    facecolor='white',    # Background color
    transparent=False     # For PNG transparency
)
# Show the plot
plt.show()