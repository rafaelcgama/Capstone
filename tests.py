# %%
import os
import pandas as pd  # library for data analysis
import numpy as np  # library to handle data in a vectorized manner
from time import sleep
from unidecode import unidecode

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from geopy.geocoders import Nominatim  # convert an address into latitude and longitude values

import requests  # library to handle requests
from lxml import html  # library to parse HTML
import json

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

# import to encode categorized features
from sklearn import preprocessing

import folium  # map rendering library

print('Libraries imported.')

# %% [markdown]
# <a id='item1'></a>
# %% [markdown]
# ## 1. Download and Explore Dataset
# %% [markdown]
# #### Load and explore the data

# %%
try:
    nyc_data = requests.get("https://cocl.us/new_york_dataset").json()
except Exception:
    with open('newyork_data.json') as json_data:
        nyc_data = json.load(json_data)
print('Data downloaded!')

# %% [markdown]
# All the relevant data is in the *features* key, which is basically a list of the neighborhoods. So, let's define a new variable that includes this data.

# %%
neighborhoods_data = nyc_data['features']

# %% [markdown]
# Let's take a look at the first item in this list.

# %%
neighborhoods_data[0]

# %% [markdown]
# #### Transform the data into a *pandas* dataframe
# %% [markdown]
# The next task is essentially transforming this data of nested Python dictionaries into a *pandas* dataframe. So let's start by creating an empty dataframe.

# %%
# define the dataframe columns
column_names = ['borough', 'neighborhood', 'latitude', 'longitude']

# instantiate the dataframe
nyc_data = pd.DataFrame(columns=column_names)

# %% [markdown]
# Take a look at the empty dataframe to confirm that the columns are as intended.

# %%
print(nyc_data)

# %% [markdown]
# Then let's loop through the data and fill the dataframe one row at a time.

# %%
for data in neighborhoods_data:
    borough = neighborhood_name = data['properties']['borough']
    neighborhood_name = data['properties']['name']

    neighborhood_latlon = data['geometry']['coordinates']
    neighborhood_lat = neighborhood_latlon[1]
    neighborhood_lon = neighborhood_latlon[0]

    nyc_data = nyc_data.append({'borough': borough,
                                'neighborhood': neighborhood_name,
                                'latitude': neighborhood_lat,
                                'longitude': neighborhood_lon
                                },
                               ignore_index=True
                               )

# %% [markdown]
# Quickly examine the resulting dataframe.

# %%
nyc_data.head()

# %% [markdown]
# And make sure that the dataset has all 5 boroughs and 306 neighborhoods.

# %%
print('The dataframe has {} boroughs and {} neighborhoods.'.format(
    len(nyc_data['borough'].unique()),
    nyc_data.shape[0]
)
)

# %% [markdown]
# Let's encode NYC's boroughs

# %%
region_encoder = preprocessing.LabelEncoder()
region_encoder.fit(nyc_data['borough'])
nyc_data.insert(1, 'borough_encoded', region_encoder.transform(nyc_data['borough']))
nyc_data.head()

# %% [markdown]
# #### Use geopy library to get the latitude and longitude values of New York City.
# %% [markdown]
# In order to define an instance of the geocoder, we need to define a user_agent. We will name our agent <em>ny_explorer</em>, as shown below.

# %%
address = 'New York City, NY'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geographical coordinate of New York City are {}, {}.'.format(latitude, longitude))

# %% [markdown]
# #### Create a map of New York with neighborhoods superimposed on top.

# %%
# create map of New York using latitude and longitude values
map_nyc = folium.Map(location=[latitude, longitude], zoom_start=10)

num_borough = nyc_data['borough_encoded'].unique()
color_range = list(range(0, len(num_borough)))
colors_array = [cm.tab10(x + 1) for x in color_range]
cat_colors = [colors.rgb2hex(i) for i in colors_array]

# add markers to map
for lat, lng, borough, neighborhood, borough_encoded in zip(nyc_data['latitude'], nyc_data['longitude'],
                                                            nyc_data['borough'], nyc_data['neighborhood'],
                                                            nyc_data['borough_encoded']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color=cat_colors[borough_encoded],
        fill=True,
        fill_color=cat_colors[borough_encoded],
        fill_opacity=0.7,
        parse_html=False).add_to(map_nyc)


# %% [markdown]
# ## 2. Explore Neighborhoods in Manhattan
# %% [markdown]
# #### Define Foursquare Credentials and Version

# %%
CLIENT_ID = os.getenv('CLIENT_ID') # your Foursquare ID
CLIENT_SECRET = os.getenv('CLIENT_SECRET') # your Foursquare Secret
VERSION = os.getenv('VERSION') # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)

# %% [markdown]
# #### Let's create a function to repeat the same process to all the neighborhoods

# %%
LIMIT = 100


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    venues_list = []
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)

        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID,
            CLIENT_SECRET,
            VERSION,
            lat,
            lng,
            radius,
            LIMIT)

        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        sleep(1)

        # return only relevant information for each nearby venue
        venues_list.append([(
            name,
            lat,
            lng,
            v['venue']['name'],
            v['venue']['location']['lat'],
            v['venue']['location']['lng'],
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['neighborhood',
                             'neighborhood_latitude',
                             'neighborhood_longitude',
                             'venue',
                             'venue_latitude',
                             'venue_longitude',
                             'venue_category'
                             ]

    return nearby_venues


# %% [markdown]
# #### Now write the code to run the above function on each neighborhood and create a new dataframe called *nyc_venues*.

# %%
# type your answer here
nyc_venues = getNearbyVenues(names=nyc_data['neighborhood'],
                             latitudes=nyc_data['latitude'],
                             longitudes=nyc_data['longitude']
                             )
nyc_venues.head()

# %% [markdown]
# #### Let's check the size of the resulting dataframe

# %%
print(nyc_venues.shape)
nyc_venues.head()

# %% [markdown]
# Let's check how many venues were returned for each neighborhood

# %%
nyc_venues.groupby('neighborhood').count().head()

# %% [markdown]
# #### Let's find out how many unique categories can be curated from all the returned venues

# %%
print('There are {} uniques categories.'.format(len(nyc_venues['venue_category'].unique())))

# %% [markdown]
# <a id='item3'></a>
# %% [markdown]
# ## 3. Analyze Each Neighborhood

# %%
# one hot encoding
nyc_onehot = pd.get_dummies(nyc_venues[['venue_category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
nyc_onehot['neighborhood'] = nyc_venues['neighborhood']

# move neighborhood column to the first column
fixed_columns = [nyc_onehot.columns[-1]] + list(nyc_onehot.columns[:-1])
nyc_onehot = nyc_onehot[fixed_columns]

print(nyc_onehot.shape)
nyc_onehot.head()

# %% [markdown]
# #### Next, let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category

# %%
nyc_grouped = nyc_onehot.groupby('neighborhood').mean().reset_index()
print(nyc_grouped.shape)
nyc_grouped.head()


# %% [markdown]
# #### Let's put that into a *pandas* dataframe
# %% [markdown]
# First, let's write a function to sort the venues in descending order.

# %%
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)

    return row_categories_sorted.index.values[0:num_top_venues]


# %% [markdown]
# Now let's create the new dataframe and display the top 10 venues for each neighborhood.

# %%
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind + 1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind + 1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['neighborhood'] = nyc_grouped['neighborhood']

for ind in np.arange(nyc_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(nyc_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()

# %% [markdown]
# <a id='item4'></a>
# %% [markdown]
# ## 4. Cluster Neighborhoods
# %% [markdown]
# Before running the *k*-means model, let's use the elbow model to selected the best K.

# %%
from yellowbrick.cluster import KElbowVisualizer

nyc_part_clustering = nyc_grouped.drop('neighborhood', 1)

# Instantiate the clustering model and visualizer
model = KMeans(random_state=42)
visualizer = KElbowVisualizer(model, k=(4, 12), metric='silhouette', timings=False)

visualizer.fit(nyc_part_clustering)  # Fit the data to the visualizer
visualizer.poof()

# %% [markdown]
# Run *k*-means to cluster the neighborhood into 5 clusters.

# %%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Use silhouette score
range_n_clusters = list(range(4, 13))

for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters)
    preds = clusterer.fit_predict(nyc_part_clustering)
    centers = clusterer.cluster_centers_

    score = silhouette_score(nyc_part_clustering, preds)
    print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))

# %%
# set number of clusters
kclusters = 10
nyc_grouped_clustering = nyc_grouped.drop('neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(nyc_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]

# %% [markdown]
# Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.

# %%
# add clustering labels
neighborhoods_venues_sorted.insert(0, 'cluster_labels', kmeans.labels_)

nyc_merged = nyc_data.copy()

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
nyc_merged = nyc_merged.join(neighborhoods_venues_sorted.set_index('neighborhood'), on='neighborhood')

print(nyc_merged.head())
nyc_merged.head()  # check the last columns!

# %% [markdown]
# Finally, let's visualize the resulting clusters

# %%
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
color_range = list(range(0, kclusters))
colors_array = [cm.tab10(x) for x in color_range]
cat_colors = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(nyc_merged['latitude'], nyc_merged['longitude'], nyc_merged['neighborhood'],
                                  nyc_merged['cluster_labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=cat_colors[cluster],
        fill=True,
        fill_color=cat_colors[cluster],
        fill_opacity=0.7).add_to(map_clusters)

map_clusters

# %% [markdown]
# <a id='item5'></a>
# %% [markdown]
# ## 5. Examine Clusters
# %% [markdown]
# Now, you can examine each cluster and determine the discriminating venue categories that distinguish each cluster. Based on the defining categories, you can then assign a name to each cluster. I will leave this exercise to you.
# %% [markdown]
# #### Cluster 1

# %%
nyc_merged.loc[nyc_merged['cluster_labels'] == 0, nyc_merged.columns[[1] + list(range(10, nyc_merged.shape[1]))]]

# %% [markdown]
# #### Cluster 2

# %%
nyc_merged.loc[nyc_merged['cluster_labels'] == 1, nyc_merged.columns[[1] + list(range(10, nyc_merged.shape[1]))]]

# %% [markdown]
# #### Cluster 3

# %%
nyc_merged.loc[nyc_merged['cluster_labels'] == 2, nyc_merged.columns[[1] + list(range(10, nyc_merged.shape[1]))]]

# %% [markdown]
# #### Cluster 4

# %%
nyc_merged.loc[nyc_merged['cluster_labels'] == 3, nyc_merged.columns[[1] + list(range(10, nyc_merged.shape[1]))]]

# %% [markdown]
# #### Cluster 5

# %%
nyc_merged.loc[nyc_merged['cluster_labels'] == 4, nyc_merged.columns[[1] + list(range(10, nyc_merged.shape[1]))]]

# %% [markdown]
# #### Cluster 6

# %%
nyc_merged.loc[nyc_merged['cluster_labels'] == 5, nyc_merged.columns[[1] + list(range(10, nyc_merged.shape[1]))]]

# %% [markdown]
# #### Cluster 7

# %%
nyc_merged.loc[nyc_merged['cluster_labels'] == 6, nyc_merged.columns[[1] + list(range(10, nyc_merged.shape[1]))]]

# %% [markdown]
# #### Cluster 8

# %%
nyc_merged.loc[nyc_merged['cluster_labels'] == 7, nyc_merged.columns[[1] + list(range(10, nyc_merged.shape[1]))]]

# %% [markdown]
# #### Cluster 9

# %%
nyc_merged.loc[nyc_merged['cluster_labels'] == 8, nyc_merged.columns[[1] + list(range(10, nyc_merged.shape[1]))]]

# %% [markdown]
# #### Cluster 10

# %%
nyc_merged.loc[nyc_merged['cluster_labels'] == 9, nyc_merged.columns[[1] + list(range(10, nyc_merged.shape[1]))]]

# %% [markdown]
# # London
# %% [markdown]
# London is composed of 32 boroughs plus the City of London that are grouped in 5 sub-regions. Each borough is composed of a few "areas" which are equivalent to the neighborhoods in New York City. So in order to create a similar dataset I will collect data from two wikipedia pages and combine the results.

# %%
url_areas = "https://en.wikipedia.org/wiki/List_of_areas_of_London"
url_regions = 'https://en.wikipedia.org/wiki/List_of_sub-regions_used_in_the_London_Plan'

session = requests.session()
session.verify = False

resp_areas = session.get(url_areas)
resp_regions = session.get(url_regions)

root_areas = html.fromstring(resp_areas.content)
root_regions = html.fromstring(resp_regions.content)


# %%
# Function to extract data from wikipedia tables
def extract_data(root):
    row_combox = root.xpath('.//table[contains(@class, "wikitable")]')
    row_combox = row_combox[0].xpath('.//tbody/tr')
    cols = [unidecode(line.text.strip()).replace(" ", "_").lower() if line.text is not None else None for line in
            list(row_combox[0])]

    data = []
    for row in row_combox[1:]:
        row_dict = {}
        for col in zip(cols, list(row)):
            row_dict[col[0]] = col[1].text_content().split('[')[0].strip()

        data.append(row_dict)

    return data


# %%
london_areas = extract_data(root_areas)
london_areas = pd.DataFrame(london_areas)
print(london_areas.shape)
london_areas.head()


# %% [markdown]
# ### Data Wrangling 
# %% [markdown]
# First, I will modify the london_areas data so there is only one borough per location. The reason for that, is because in order to assign a region to a location, I need to know its borough. Let's use the geopy library to search the location and use the borough it returns. 
# %% [markdown]
# Remove parentheses from location  

# %%
def remove_parentheses(row):
    location = row['location']
    if '(' in location:
        location = location.split('(')[0].strip()
    return location


# %%
london_areas['location'] = london_areas.apply(remove_parentheses, axis=1)
print(london_areas.shape)
london_areas.head()

# %% [markdown]
# Select locations with more than one borough

# %%
borough_comma = london_areas.loc[london_areas['london_borough'].str.contains(', | &')]
print(borough_comma.shape)
borough_comma.head()


# %%
def select_neighborhood(row):
    area = row['location']
    borough = row['london_borough']
    nomi = Nominatim(user_agent='london_app')
    result = nomi.geocode(f'{area}, London, United Kingdom')

    for line in result[0].split(','):
        if 'Borough of' in line:
            borough = line.split('of')[-1].strip()
            break

    return borough


# %%
borough_comma.loc[:, 'london_borough'] = borough_comma.apply(select_neighborhood, axis=1)
borough_comma.loc[borough_comma['london_borough'].str.contains(', | &')]

# %% [markdown]
# As we can see there are still some locations where it wasn't possible to update the neighborhood using geopy so those I will research online and fix by hand.  

# %%
indexes = borough_comma.loc[borough_comma['london_borough'].str.contains(', | &')].index
new_boroughs = ['Lambeth', 'Camden', 'Bromley', 'City', 'Greenwich', 'Bromley']

borough_comma.loc[indexes, 'london_borough'] = new_boroughs
print(borough_comma)

# %%
london_areas.loc[borough_comma.index, 'london_borough'] = borough_comma['london_borough']
print(london_areas.iloc[borough_comma.index].shape)
london_areas.iloc[borough_comma.index].head()

# %% [markdown]
# Convert OS Grid (British National Grid References) to Latitude & Longitude

# %%
from OSGridConverter import grid2latlong


def grid_conversion(row):
    grid = row['os_grid_ref'].strip()
    latitude = None
    longitude = None
    if len(grid):
        coord = grid2latlong(grid)
        latitude = coord.latitude
        longitude = coord.longitude

    else:
        nomi = Nominatim(user_agent='london_app')
        location = row['location']
        borough = row['london_borough']
        result = nomi.geocode(f'{location}, {borough}, London, United Kingdom')

        if result is not None:
            latitude = result[1][0]
            longitude = result[1][1]

    return latitude, longitude


# %%
london_areas[['latitude', 'longitude']] = ''

london_areas[['latitude', 'longitude']] = london_areas.apply(grid_conversion, axis=1, result_type='expand')
print(london_areas.shape)
london_areas.head()

# %% [markdown]
# Rename borough "City" to "City of London"

# %%
london_areas['london_borough'] = london_areas['london_borough'].str.replace('City', 'City of London')
london_areas.head(10)

# %% [markdown]
# Fix problematic locations

# %%
london_areas.loc[(london_areas['location'] == 'Brompton') | (london_areas['location'] == 'Sudbury') | (
        london_areas['location'] == 'Somerstown')]

# %%
london_areas.loc[68, 'london_borough'] = 'Kensington and Chelsea'
london_areas.loc[411, 'location'] = 'Somers Town'
london_areas.loc[448, 'london_borough'] = 'Brent'
london_areas.iloc[[68, 411, 448]]

# %% [markdown]
# Now that we fixed the dataset, let's check if we got 33 unique boroughs, which is London's official number of boroughs.

# %%
len(london_areas['london_borough'].unique())

# %% [markdown]
# We have 3 more boroughs than we should have. In order to find out which ones are wrong, I will download an oficial government dataset containing information about London's boroughs and will compare it with our list.

# %%
official_boroughs = pd.read_csv(
    'https://data.london.gov.uk/download/london-borough-profiles/c1693b82-68b1-44ee-beb2-3decf17dc1f8/london-borough-profiles.csv',
    engine='python', encoding='latin1')
official_boroughs.head()

# %%
official_boroughs.dropna(inplace=True)
official_areas = official_boroughs['Area_name']
official_areas.head()

# %%
# Extra boroughs
myareas = pd.Series(london_areas['london_borough'].unique())

myareas.loc[~myareas.isin(official_areas)]

# %% [markdown]
# Fix problematic boroughs

# %%
london_areas.loc[(london_areas['london_borough'] == 'Dartford')]

# %%
london_areas.loc[133, 'london_borough'] = 'Bexley'
london_areas['london_borough'] = london_areas['london_borough'].str.replace('Camden and Islington', 'Camden')
london_areas['london_borough'] = london_areas['london_borough'].str.replace('Haringey and Barnet', 'Haringey')

len(london_areas['london_borough'].unique())

# %%
# Rename columns
london_areas.rename(columns={'location': 'neighborhood', 'london_borough': 'borough'}, inplace=True)

# %% [markdown]
# Now let's collect the assign the locations to its respective regions 

# %%
london_regions = extract_data(root_regions)
london_regions = pd.DataFrame(london_regions)
print(london_regions.shape)
london_regions.head()

# %% [markdown]
# Spread the dataset

# %%
london_regions = pd.DataFrame(london_regions['london_boroughs'].str.split(',').tolist(),
                              index=london_regions['sub-region']).stack()
london_regions = london_regions.reset_index()[['sub-region', 0]]

london_regions = london_regions.apply(lambda x: x.str.strip())

london_regions.rename(columns={'sub-region': 'sub_region', 0: 'borough'}, inplace=True)
london_regions.head()

# %% [markdown]
# Now, let's merge the tables

# %%
london_data = pd.merge(london_regions, london_areas[['borough', 'neighborhood', 'latitude', 'longitude']], on='borough',
                       how='inner').reset_index(drop=True)
london_data.head()

# %%
# Check for nulls 
london_data.isnull().sum()

# %%
num_neighborhoods = london_data.shape[0]
num_boroughs = len(london_data['borough'].unique())
print(f'There are {num_boroughs} boroughs and {num_neighborhoods} neighborhoods in London.')

# %% [markdown]
# Now that the dataset is clean, let's check for duplicates in the coordinates. The reason this wasn't done before is because I wanted to fix the inconsistencies beforehand so I have the correct data to search for the correct coordinates in case I have to fix something.

# %%
duplicated_coord = london_data.loc[london_data.duplicated(subset=['latitude', 'longitude'], keep=False)]
print(duplicated_coord.shape)
duplicated_coord.head()

# %%
print(
    f"There are {duplicated_coord.shape[0]} neighborhoods that have potentially bad coordinates so I will use geopy to search for all of them")

# %%
nomi = Nominatim(user_agent='london')
nomi.geocode(f'Bloomsbury, Camden, London, United Kingdom')[1]


# %%
# Get coordinates 
def get_coordinates(row):
    neighborhood = row['neighborhood']
    borough = row['borough']
    nomi = Nominatim(user_agent='london')
    result = nomi.geocode(f'{neighborhood}, {borough}, London, United Kingdom')

    latitude = None
    longitude = None

    if result is not None:
        latitude = result[1][0]
        longitude = result[1][1]

    return latitude, longitude


# %%
duplicated_coord[['latitude', 'longitude']] = duplicated_coord.apply(get_coordinates, axis=1, result_type='expand')
duplicated_coord

# %% [markdown]
# Look for, and fix nulls values in the new coordinates

# %%
duplicated_coord.loc[duplicated_coord.isnull().any(1)]

# %% [markdown]
# After checking on Google, the original coordinates for those two neighborhoods were correct so I will keep them in the dataset and update the new ones.

# %%
index2update = np.setdiff1d(duplicated_coord.index, duplicated_coord.loc[duplicated_coord.isnull().any(1)].index)
london_data.loc[index2update, ['latitude', 'longitude']] = duplicated_coord.loc[index2update, ['latitude', 'longitude']]

print(
    f"There are {london_data.duplicated(subset=['latitude', 'longitude'], keep=False).sum()} duplicate coordinates left.")

# %% [markdown]
# #### Change the coordinates columns data type

# %%
london_data['latitude'] = london_data['latitude'].astype('float64')
london_data['longitude'] = london_data['longitude'].astype('float64')
london_data.info()

# %% [markdown]
# Use geopy library to get the latitude and longitude values of London.

# %%
address = 'London, England'

geolocator = Nominatim(user_agent="london_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geographical coordinate of London are {}, {}.'.format(latitude, longitude))

# %% [markdown]
# Encode "sub_region" column

# %%
from sklearn import preprocessing

region_encoder = preprocessing.LabelEncoder()
region_encoder.fit(london_data['sub_region'])
region_encoder.classes_

# %%
london_data.insert(1, 'sub_region_encoded', region_encoder.transform(london_data['sub_region']))
london_data.head()

# %% [markdown]
# #### Create a map of London with neighborhoods superimposed on top.

# %%
# create map of London using latitude and longitude values
map_london = folium.Map(location=[latitude, longitude], zoom_start=10)

# set color scheme for the sub regions
num_regions = london_data['sub_region_encoded'].unique()
color_range = list(range(0, len(num_regions)))
colors_array = [cm.tab10(x + 1) for x in color_range]
cat_colors = [colors.rgb2hex(i) for i in colors_array]

# add markers to map
for lat, lng, borough, neighborhood, region in zip(london_data['latitude'], london_data['longitude'],
                                                   london_data['borough'], london_data['neighborhood'],
                                                   london_data['sub_region_encoded']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color=cat_colors[region],
        fill=True,
        fill_color=cat_colors[region],
        fill_opacity=0.7,
        parse_html=False).add_to(map_london)

map_london

# %% [markdown]
# Next, we are going to start utilizing the Foursquare API to explore the neighborhoods and segment them.
# %% [markdown]
# ## 2. Explore Neighborhoods in London

# %%
london_venues = getNearbyVenues(names=london_data['neighborhood'],
                                latitudes=london_data['latitude'],
                                longitudes=london_data['longitude']
                                )

# %% [markdown]
# #### Let's check the size of the resulting dataframe

# %%
print(london_venues.shape)
london_venues.head()
london_venues.shape

# %% [markdown]
# Let's check how many venues were returned for each neighborhood

# %%
london_venues.groupby('neighborhood').count()
london_venues.groupby('neighborhood').count().shape

# %% [markdown]
# #### Let's find out how many unique categories can be curated from all the returned venues

# %%
print('There are {} uniques categories.'.format(len(london_venues['venue_category'].unique())))

# %% [markdown]
# <a id='item3'></a>
# %% [markdown]
# ## 3. Analyze Each Neighborhood

# %%
# one hot encoding
london_onehot = pd.get_dummies(london_venues[['venue_category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
london_onehot['neighborhood'] = london_venues['neighborhood']

# move neighborhood column to the first column
fixed_columns = [london_onehot.columns[-1]] + list(london_onehot.columns[:-1])
london_onehot = london_onehot[fixed_columns]

print(london_onehot.shape)
london_onehot.head()

# %% [markdown]
# #### Next, let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category

# %%
london_grouped = london_onehot.groupby('neighborhood').mean().reset_index()
print(london_grouped.shape)
london_grouped.head()

# %% [markdown]
# #### Let's print each neighborhood along with the top 5 most common venues

# %%
num_top_venues = 10

for hood in london_grouped['neighborhood']:
    print("----" + hood + "----")
    temp = london_grouped[london_grouped['neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue', 'freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')

# %% [markdown]
# #### Let's put that into a *pandas* dataframe and display the top 10 venues for each neighborhood

# %%
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind + 1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind + 1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['neighborhood'] = london_grouped['neighborhood']

for ind in np.arange(london_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(london_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()
neighborhoods_venues_sorted.shape

# %% [markdown]
# ## 4. Cluster Neighborhoods
# %% [markdown]
# Run *k*-means to cluster the neighborhood into 5 clusters.

# %%
london_part_clustering = london_grouped.drop('neighborhood', 1)

# Instantiate the clustering model and visualizer
model = KMeans(random_state=42)
visualizer = KElbowVisualizer(model, k=(4, 11), metric='silhouette', timings=False)

visualizer.fit(london_part_clustering)  # Fit the data to the visualizer
visualizer.poof()

# %%
# set number of clusters
kclusters = 6

london_grouped_clustering = london_grouped.drop('neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=42).fit(london_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]

# %% [markdown]
# Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.
# %% [markdown]
# Before the merge let's check if both tables have all the same neighborhoods

# %%
london_data.loc[~london_data['neighborhood'].isin(london_data['neighborhood'])]

# %%
# add clustering labels
neighborhoods_venues_sorted.insert(0, 'cluster_labels', kmeans.labels_)

london_merged = london_data.copy()

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
london_merged = london_merged.iloc[:, 3:].join(neighborhoods_venues_sorted.set_index('neighborhood'), on='neighborhood',
                                               how='inner')

london_merged.head()  # check the last columns!
london_merged.shape

# %%
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
color_range = list(range(0, kclusters))
colors_array = [cm.tab10(x) for x in color_range]
cat_colors = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(london_merged['latitude'], london_merged['longitude'], london_merged['neighborhood'],
                                  london_merged['cluster_labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=cat_colors[cluster],
        fill=True,
        fill_color=cat_colors[cluster],
        fill_opacity=0.7).add_to(map_clusters)

map_clusters

# %% [markdown]
# <a id='item5'></a>
# %% [markdown]
# ## 5. Examine Clusters
# %% [markdown]
# Now, you can examine each cluster and determine the discriminating venue categories that distinguish each cluster. Based on the defining categories, you can then assign a name to each cluster. I will leave this exercise to you.
# %% [markdown]
# #### Cluster 1

# %%
london_merged.loc[
    london_merged['cluster_labels'] == 0, london_merged.columns[[1] + list(range(6, london_merged.shape[1]))]]

# %% [markdown]
# #### Cluster 2

# %%
london_merged.loc[
    london_merged['cluster_labels'] == 1, london_merged.columns[[1] + list(range(6, london_merged.shape[1]))]]

# %% [markdown]
# #### Cluster 3

# %%
london_merged.loc[
    london_merged['cluster_labels'] == 2, london_merged.columns[[1] + list(range(6, london_merged.shape[1]))]]

# %% [markdown]
# #### Cluster 4

# %%
london_merged.loc[
    london_merged['cluster_labels'] == 3, london_merged.columns[[1] + list(range(6, london_merged.shape[1]))]]

# %% [markdown]
# #### Cluster 5

# %%
london_merged.loc[
    london_merged['cluster_labels'] == 4, london_merged.columns[[1] + list(range(6, london_merged.shape[1]))]]

# %% [markdown]
# #### Cluster 6

# %%
london_merged.loc[
    london_merged['cluster_labels'] == 5, london_merged.columns[[1] + list(range(6, london_merged.shape[1]))]]

print("Finally")
