import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

df = pd.read_excel('pre_1910_hydroshare.xlsx')

print("Missing values before cleaning:")
print(df.isnull().sum())

df_clean = df.dropna(subset=['latitude', 'longitude'])

# Create figure with map projection
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Extract coordinates
lon = df_clean['longitude'].values
lat = df_clean['latitude'].values

if len(lon) == len(lat) and len(lon) > 0:
    heatmap, xedges, yedges = np.histogram2d(lon, lat, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    im = ax.imshow(heatmap.T, extent=extent,
                  origin='lower', cmap='hot',
                  alpha=0.7, transform=ccrs.PlateCarree())
    
    plt.colorbar(im, ax=ax, label='Density')
    plt.title('Data Heatmap')
else:
    print("Error: Coordinate arrays still mismatched after cleaning")
    print(f"Final lon length: {len(lon)}")
    print(f"Final lat length: {len(lat)}")


# Optional:
# ax.scatter(lon, lat, s=10, c='red', 
#           alpha=0.3, transform=ccrs.PlateCarree())

# Set map bounds
ax.set_extent([lon.min()-1, lon.max()+1, 
              lat.min()-1, lat.max()+1])

# Add grid and title
ax.gridlines(draw_labels=True)
plt.title('Geographical Heatmap')
plt.savefig(
    'heatmap.pdf',  # File name
    dpi=600,          # Image resolution (300-600 for publications)
    bbox_inches='tight',  # Prevent cropped borders
    facecolor='white',    # Background color
    transparent=False     # For PNG transparency
)
plt.show()