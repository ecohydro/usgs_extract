import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.metrics import pairwise_distances
from scipy.linalg import eigh

def diffusion_map(coords, epsilon=0.1, n_components=2):
    """
    Create a diffusion map from coordinates
    :param coords: Numpy array of shape (n_samples, 2) [lat, long]
    :param epsilon: Kernel bandwidth parameter
    :param n_components: Number of diffusion components to return
    :return: Tuple of (eigenvalues, eigenvectors)
    """
    # 1. Compute pairwise distances
    dists = pairwise_distances(coords, metric='euclidean')
    
    # 2. Construct Gaussian kernel matrix
    K = np.exp(-dists**2 / epsilon**2)
    
    # 3. Compute normalized kernel matrix
    D = np.diag(1 / K.sum(axis=1))
    L = D @ K  # Normalized kernel
    
    # 4. Compute eigenvalues and eigenvectors
    vals, vecs = eigh(L)
    
    # Sort in descending order
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    return vals[:n_components], vecs[:, :n_components]

# Load and preprocess data
df = pd.read_csv('metadata.csv')
df = df.dropna(subset=['Actual_Latitude', 'Actual_Longitude'])
df = df[pd.to_numeric(df['Actual_Latitude'], errors='coerce').notnull()]
df = df[pd.to_numeric(df['Actual_Longitude'], errors='coerce').notnull()]
coords = df[['Actual_Latitude', 'Actual_Longitude']].values

# Normalize coordinates
coords = (coords - coords.mean(axis=0)) / coords.std(axis=0)

# Compute diffusion map
epsilon = 0.5  # Adjust based on your data scale
vals, vecs = diffusion_map(coords, epsilon=epsilon)

# Create visualization
fig = plt.figure(figsize=(18, 8))

# Geographic plot
ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree())
ax1.add_feature(cfeature.LAND)
ax1.add_feature(cfeature.COASTLINE)
sc1 = ax1.scatter(df['Actual_Longitude'], df['Actual_Latitude'], c=vecs[:, 1], 
                cmap='viridis', s=10, transform=ccrs.PlateCarree())
plt.colorbar(sc1, ax=ax1, label='Diffusion Component 1')
ax1.set_title('Geographical Space')

# Diffusion space plot
ax2 = fig.add_subplot(122)
sc2 = ax2.scatter(vecs[:, 0], vecs[:, 1], c=vecs[:, 1], cmap='viridis', s=50)
plt.colorbar(sc2, ax=ax2, label='Diffusion Component 1')
ax2.set_xlabel('Component 0')
ax2.set_ylabel('Component 1')
ax2.set_title('Diffusion Space')

plt.tight_layout()
plt.savefig(
    'diffusion.pdf',  
    dpi=600,          
    bbox_inches='tight',  
    facecolor='white',    
    transparent=False     
)
plt.show()