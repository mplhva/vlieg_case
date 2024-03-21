# Pad van Anaconda
# cd OneDrive\Bureaublad\Data_Science\Visual_Analytics\Presentaties\Presentatie_3

# Streamlit:
# streamlit run app_flight.py

import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import webbrowser
from datetime import datetime 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from folium.plugins import MarkerCluster

# Lees het CSV-bestand in een DataFrame, met scheidingsteken ';'
df1 = pd.read_csv('airports-extended-clean.csv', sep=';')

# Verwijder duplicaten uit de DataFrame
df1 = df1.drop_duplicates()

# Verwijder vervolgens eventuele ontbrekende waarden uit de DataFrame
df1 = df1.dropna()

# Define a dictionary mapping countries to their continents
country_to_continent = {
    'Afghanistan': 'Asia',
    'Albania': 'Europe',
    'Algeria': 'Africa',
    'American Samoa': 'Oceania',
    'Angola': 'Africa',
    'Anguilla': 'North America',
    'Antarctica': 'Antarctica',
    'Antigua and Barbuda': 'North America',
    'Argentina': 'South America',
    'Armenia': 'Asia',
    'Aruba': 'North America',
    'Ashmore and Cartier Islands': 'Oceania',
    'Australia': 'Oceania',
    'Austria': 'Europe',
    'Azerbaijan': 'Asia',
    'Bahamas': 'North America',
    'Bahrain': 'Asia',
    'Bangladesh': 'Asia',
    'Barbados': 'North America',
    'Belarus': 'Europe',
    'Belgium': 'Europe',
    'Belize': 'North America',
    'Benin': 'Africa',
    'Bermuda': 'North America',
    'Bhutan': 'Asia',
    'Bolivia': 'South America',
    'Bosnia and Herzegovina': 'Europe',
    'Botswana': 'Africa',
    'Brazil': 'South America',
    'British Indian Ocean Territory': 'Asia',
    'British Virgin Islands': 'North America',
    'Brunei': 'Asia',
    'Bulgaria': 'Europe',
    'Burkina Faso': 'Africa',
    'Burma': 'Asia',
    'Burundi': 'Africa',
    'Cambodia': 'Asia',
    'Cameroon': 'Africa',
    'Canada': 'North America',
    'Cape Verde': 'Africa',
    'Cayman Islands': 'North America',
    'Central African Republic': 'Africa',
    'Chad': 'Africa',
    'Chile': 'South America',
    'China': 'Asia',
    'Christmas Island': 'Asia',
    'Cocos (Keeling) Islands': 'Asia',
    'Colombia': 'South America',
    'Comoros': 'Africa',
    'Congo (Brazzaville)': 'Africa',
    'Congo (Kinshasa)': 'Africa',
    'Cook Islands': 'Oceania',
    'Costa Rica': 'North America',
    "Cote d'Ivoire": 'Africa',
    'Croatia': 'Europe',
    'Cuba': 'North America',
    'Cyprus': 'Asia',
    'Czech Republic': 'Europe',
    'Denmark': 'Europe',
    'Djibouti': 'Africa',
    'Dominica': 'North America',
    'Dominican Republic': 'North America',
    'East Timor': 'Asia',
    'Ecuador': 'South America',
    'Egypt': 'Africa',
    'El Salvador': 'North America',
    'Equatorial Guinea': 'Africa',
    'Eritrea': 'Africa',
    'Estonia': 'Europe',
    'Ethiopia': 'Africa',
    'Falkland Islands': 'South America',
    'Faroe Islands': 'Europe',
    'Fiji': 'Oceania',
    'Finland': 'Europe',
    'France': 'Europe',
    'French Guiana': 'South America',
    'French Polynesia': 'Oceania',
    'Gabon': 'Africa',
    'Gambia': 'Africa',
    'Georgia': 'Europe',
    'Germany': 'Europe',
    'Ghana': 'Africa',
    'Gibraltar': 'Europe',
    'Greece': 'Europe',
    'Greenland': 'North America',
    'Grenada': 'North America',
    'Guadeloupe': 'North America',
    'Guam': 'Oceania',
    'Guatemala': 'North America',
    'Guernsey': 'Europe',
    'Guinea': 'Africa',
    'Guinea-Bissau': 'Africa',
    'Guyana': 'South America',
    'Haiti': 'North America',
    'Honduras': 'North America',
    'Hong Kong': 'Asia',
    'Hungary': 'Europe',
    'Iceland': 'Europe',
    'India': 'Asia',
    'Indonesia': 'Asia',
    'Iran': 'Asia',
    'Iraq': 'Asia',
    'Ireland': 'Europe',
    'Isle of Man': 'Europe',
    'Israel': 'Asia',
    'Italy': 'Europe',
    'Jamaica': 'North America',
    'Japan': 'Asia',
    'Jersey': 'Europe',
    'Johnston Atoll': 'Oceania',
    'Jordan': 'Asia',
    'Juan de Nova Island': 'Africa',
    'Kazakhstan': 'Asia',
    'Kenya': 'Africa',
    'Kiribati': 'Oceania',
    'Kuwait': 'Asia',
    'Kyrgyzstan': 'Asia',
    'Laos': 'Asia',
    'Latvia': 'Europe',
    'Lebanon': 'Asia',
    'Lesotho': 'Africa',
    'Liberia': 'Africa',
    'Libya': 'Africa',
    'Lithuania': 'Europe',
    'Luxembourg': 'Europe',
    'Macau': 'Asia',
    'Macedonia': 'Europe',
    'Madagascar': 'Africa',
    'Malawi': 'Africa',
    'Malaysia': 'Asia',
    'Maldives': 'Asia',
    'Mali': 'Africa',
    'Malta': 'Europe',
    'Marshall Islands': 'Oceania',
    'Martinique': 'North America',
    'Mauritania': 'Africa',
    'Mauritius': 'Africa',
    'Mayotte': 'Africa',
    'Mexico': 'North America',
    'Micronesia': 'Oceania',
    'Midway Islands': 'North America',
    'Moldova': 'Europe',
    'Monaco': 'Europe',
    'Mongolia': 'Asia',
    'Montenegro': 'Europe',
    'Montserrat': 'North America',
    'Morocco': 'Africa',
    'Mozambique': 'Africa',
    'Myanmar': 'Asia',
    'Namibia': 'Africa',
    'Nauru': 'Oceania',
    'Nepal': 'Asia',
    'Netherlands': 'Europe',
    'Netherlands Antilles': 'North America',
    'New Caledonia': 'Oceania',
    'New Zealand': 'Oceania',
    'Nicaragua': 'North America',
    'Niger': 'Africa',
    'Nigeria': 'Africa',
    'Niue': 'Oceania',
    'Norfolk Island': 'Oceania',
    'North Korea': 'Asia',
    'Northern Mariana Islands': 'Oceania',
    'Norway': 'Europe',
    'Oman': 'Asia',
    'Pakistan': 'Asia',
    'Palau': 'Oceania',
    'Palestine': 'Asia',
    'Panama': 'North America',
    'Papua New Guinea': 'Oceania',
    'Paraguay': 'South America',
    'Peru': 'South America',
    'Philippines': 'Asia',
    'Poland': 'Europe',
    'Portugal': 'Europe',
    'Puerto Rico': 'North America',
    'Qatar': 'Asia',
    'Reunion': 'Africa',
    'Romania': 'Europe',
    'Russia': 'Europe',
    'Rwanda': 'Africa',
    'Saint Helena': 'Africa',
    'Saint Kitts and Nevis': 'North America',
    'Saint Lucia': 'North America',
    'Saint Pierre and Miquelon': 'North America',
    'Saint Vincent and the Grenadines': 'North America',
    'Samoa': 'Oceania',
    'Sao Tome and Principe': 'Africa',
    'Saudi Arabia': 'Asia',
    'Senegal': 'Africa',
    'Serbia': 'Europe',
    'Seychelles': 'Africa',
    'Sierra Leone': 'Africa',
    'Singapore': 'Asia',
    'Slovakia': 'Europe',
    'Slovenia': 'Europe',
    'Solomon Islands': 'Oceania',
    'Somalia': 'Africa',
    'South Africa': 'Africa',
    'South Georgia and the Islands': 'South America',
    'South Korea': 'Asia',
    'South Sudan': 'Africa',
    'Spain': 'Europe',
    'Sri Lanka': 'Asia',
    'Sudan': 'Africa',
    'Suriname': 'South America',
    'Svalbard': 'Europe',
    'Swaziland': 'Africa',
    'Sweden': 'Europe',
    'Switzerland': 'Europe',
    'Syria': 'Asia',
    'Taiwan': 'Asia',
    'Tajikistan': 'Asia',
    'Tanzania': 'Africa',
    'Thailand': 'Asia',
    'Togo': 'Africa',
    'Tonga': 'Oceania',
    'Trinidad and Tobago': 'North America',
    'Tunisia': 'Africa',
    'Turkey': 'Asia',
    'Turkmenistan': 'Asia',
    'Turks and Caicos Islands': 'North America',
    'Tuvalu': 'Oceania',
    'Uganda': 'Africa',
    'Ukraine': 'Europe',
    'United Arab Emirates': 'Asia',
    'United Kingdom': 'Europe',
    'United States': 'North America',
    'Uruguay': 'South America',
    'Uzbekistan': 'Asia',
    'Vanuatu': 'Oceania',
    'Venezuela': 'South America',
    'Vietnam': 'Asia',
    'Virgin Islands': 'North America',
    'Wake Island': 'Oceania',
    'Wallis and Futuna': 'Oceania',
    'West Bank': 'Asia',
    'Western Sahara': 'Africa',
    'Yemen': 'Asia',
    'Zambia': 'Africa',
    'Zimbabwe': 'Africa'
}

# Map the countries to continents using the dictionary
df1['Continent'] = df1['Country'].map(country_to_continent)

def plot_mean_values(df):
    """
    Plot de gemiddelde waarden van de kolommen '3d Latitude', '3d Longitude', '3d Altitude M', '3d Altitude Ft' 
    en 'TRUE AIRSPEED (derived)' per continent in een staafdiagram.

    Parameters:
    df (DataFrame): DataFrame met de gegevens van vluchten, inclusief 3d breedtegraad, 3d lengtegraad, 3d hoogte, 
    True Airspeed en continent.
    """
    # Controleer de daadwerkelijke kolomnamen in het DataFrame
    columns_to_check = ['[3d Altitude M]', '[3d Altitude Ft]', 'TRUE AIRSPEED (derived)']
    missing_columns = [col for col in columns_to_check if col not in df.columns]

    if missing_columns:
        st.error(f"De volgende kolommen ontbreken in het DataFrame: {missing_columns}")
        return

    # Verwijder rijen met ontbrekende waarden in de relevante kolommen
    df = df.dropna(subset=columns_to_check)

    # Converteer kolommen naar numerieke gegevenstypen indien nodig
    numeric_columns = ['[3d Altitude M]', '[3d Altitude Ft]', 'TRUE AIRSPEED (derived)']
    for col in numeric_columns:
        if df[col].dtype != 'float64':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Controleer opnieuw of de gegevenstypen juist zijn na conversie
    for col in columns_to_check:
        if df[col].dtype != 'float64':
            st.error(f"De kolom '{col}' kon niet worden geconverteerd naar numeriek gegevenstype.")
            return

    # Filter outliers met behulp van de IQR-methode
    Q1 = df[columns_to_check].quantile(0.25)
    Q3 = df[columns_to_check].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[~((df[columns_to_check] < lower_bound) | (df[columns_to_check] > upper_bound)).any(axis=1)]

    # Bereken het gemiddelde van de relevante kolommen
    mean_values = df[columns_to_check].mean()

    # Maak een staafdiagram van het gemiddelde van de relevante kolommen
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_values.plot(kind='bar', ax=ax)
    ax.set_xlabel('Kolomnamen')
    ax.set_ylabel('Gemiddelde Waarden')
    ax.set_title('Gemiddelde waarden van vluchtgegevens (zonder outliers)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y')
    st.pyplot(fig)

    
def plot_mean_Altitude_by_continent(df):
    """
    Plot het gemiddelde van longituden van luchthavens per continent in een staafdiagram.

    Parameters:
    df (DataFrame): DataFrame met de gegevens van luchthavens, inclusief breedtegraad, lengtegraad en continent.
    """
    # Converteer de 'Longitude'-kolom naar numeriek gegevenstype
    df['Altitude'] = pd.to_numeric(df['Altitude'], errors='coerce')
    
    # Verwijder rijen met ontbrekende waarden in de 'Longitude'-kolom
    df = df.dropna(subset=['Altitude'])

    # Bereken het gemiddelde van longituden per continent
    mean_longitude_by_continent = df.groupby('Continent')['Altitude'].mean()

    # Maak een staafdiagram van het gemiddelde van longituden per continent
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_longitude_by_continent.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_xlabel('Continent')
    ax.set_ylabel('Gemiddelde Altitude')
    ax.set_title('Gemiddelde Altitude van luchthavens per continent')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y')
    st.pyplot(fig)
    
def plot_mean_Longitude_by_continent(df):
    """
    Plot het gemiddelde van longituden van luchthavens per continent in een staafdiagram.

    Parameters:
    df (DataFrame): DataFrame met de gegevens van luchthavens, inclusief breedtegraad, lengtegraad en continent.
    """
    # Converteer de 'Longitude'-kolom naar numeriek gegevenstype
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    
    # Verwijder rijen met ontbrekende waarden in de 'Longitude'-kolom
    df = df.dropna(subset=['Longitude'])

    # Bereken het gemiddelde van longituden per continent
    mean_longitude_by_continent = df.groupby('Continent')['Longitude'].mean()

    # Maak een staafdiagram van het gemiddelde van longituden per continent
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_longitude_by_continent.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_xlabel('Continent')
    ax.set_ylabel('Gemiddelde Longitude')
    ax.set_title('Gemiddelde Longitude van luchthavens per continent')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y')
    st.pyplot(fig)
    
def plot_mean_Latitude_by_continent(df):
    """
    Plot het gemiddelde van longituden van luchthavens per continent in een staafdiagram.

    Parameters:
    df (DataFrame): DataFrame met de gegevens van luchthavens, inclusief breedtegraad, lengtegraad en continent.
    """
    # Converteer de 'Longitude'-kolom naar numeriek gegevenstype
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    
    # Verwijder rijen met ontbrekende waarden in de 'Longitude'-kolom
    df = df.dropna(subset=['Latitude'])

    # Bereken het gemiddelde van longituden per continent
    mean_longitude_by_continent = df.groupby('Continent')['Latitude'].mean()

    # Maak een staafdiagram van het gemiddelde van longituden per continent
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_longitude_by_continent.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_xlabel('Continent')
    ax.set_ylabel('Gemiddelde Latitude')
    ax.set_title('Gemiddelde Latitude van luchthavens per continent')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y')
    st.pyplot(fig)

# Function to plot line charts
def plot_line_charts(df, show_arrivals=True, show_departures=True, show_net=True):
    # Filter the dataset for the desired date range (01/01/2019 to 31/12/2020)
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2020, 12, 31)
    df_filtered = df[(df['STD'] >= start_date) & (df['STD'] <= end_date)]

    # Create two separate DataFrames for arrivals (L) and departures (S) airplanes
    df_arrivals = df_filtered[df_filtered['LSV'] == 'L']
    df_departures = df_filtered[df_filtered['LSV'] == 'S']

    # Count the number of arriving and departing airplanes per date
    arrival_counts = df_arrivals['STD'].value_counts().sort_index()
    departure_counts = df_departures['STD'].value_counts().sort_index()

    # Calculate the net number of airplanes on each day (Arrivals - Departures)
    net_counts = arrival_counts.sub(departure_counts, fill_value=0)

    # Create a Streamlit app
    st.title('Number of airplanes arrived and departed (2019-2020)')

    # Add a search bar for the charts
    search_date = st.text_input("Search for a date (YYYY-MM-DD):", "")
    if search_date:
        try:
            search_date = datetime.strptime(search_date, '%Y-%m-%d')
            if start_date <= search_date <= end_date:
                selected_date = search_date
            else:
                st.warning("Please enter a date within the range (01-01-2019 to 31-12-2020).")
        except ValueError:
            st.warning("Please enter a valid date format (YYYY-MM-DD).")

    # Add a slider to select the date
    if 'selected_date' not in locals():
        selected_date = start_date
    selected_date = st.slider('Select a date', min_value=start_date, max_value=end_date, value=selected_date)

    # Create subplots for arriving and departing airplanes
    fig, axs = plt.subplots(3, 1, figsize=(15, 18))

    # Plot the line chart for arriving airplanes if show_arrivals is True
    if show_arrivals:
        axs[0].plot(arrival_counts.index, arrival_counts.values, label='Arrived (L)', color='blue')
        axs[0].axvline(x=selected_date, color='r', linestyle='--', label='Selected date')
        axs[0].set_title('Number of arriving airplanes (2019-2020)')
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel('Number of airplanes')
        axs[0].legend()
        axs[0].grid(True)

        # Show the number of arriving airplanes on the selected date
        selected_arrivals = arrival_counts[selected_date] if selected_date in arrival_counts.index else 0
        st.text(f"Number of arriving airplanes on {selected_date.strftime('%Y-%m-%d')}: {selected_arrivals}")
    else:
        axs[0].clear()

    # Plot the line chart for departing airplanes if show_departures is True
    if show_departures:
        axs[1].plot(departure_counts.index, departure_counts.values, label='Departed (S)', color='orange')
        axs[1].axvline(x=selected_date, color='r', linestyle='--', label='Selected date')
        axs[1].set_title('Number of departing airplanes (2019-2020)')
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel('Number of airplanes')
        axs[1].legend()
        axs[1].grid(True)

        # Show the number of departing airplanes on the selected date
        selected_departures = departure_counts[selected_date] if selected_date in departure_counts.index else 0
        st.text(f"Number of departing airplanes on {selected_date.strftime('%Y-%m-%d')}: {selected_departures}")
    else:
        axs[1].clear()

    # Plot the line chart for net airplanes (Arrivals - Departures) if show_net is True
    if show_net:
        axs[2].plot(net_counts.index, net_counts.values, label='Net (L - S)', color='green')
        axs[2].axvline(x=selected_date, color='r', linestyle='--', label='Selected date')
        axs[2].set_title('Net number of airplanes (2019-2020)')
        axs[2].set_xlabel('Date')
        axs[2].set_ylabel('Number of airplanes')
        axs[2].legend()
        axs[2].grid(True)

        # Calculate the net number of airplanes on the selected date
        selected_net = net_counts[selected_date] if selected_date in net_counts.index else 0
        st.text(f"Net number of airplanes on {selected_date.strftime('%Y-%m-%d')}: {selected_net}")
    else:
        axs[2].clear()

    # Improve layout
    plt.tight_layout()

    # Show the subplots based on the status of checkboxes
    show_arrivals_checkbox = st.checkbox("Show Arrivals", value=True)
    show_departures_checkbox = st.checkbox("Show Departures", value=True)
    show_net_checkbox = st.checkbox("Show Net", value=True)

    if not show_arrivals_checkbox:
        axs[0].clear()
    if not show_departures_checkbox:
        axs[1].clear()
    if not show_net_checkbox:
        axs[2].clear()

    # Show the subplots
    st.pyplot(fig)

# Load the selected file into a DataFrame
selected_file = st.selectbox("Select a file", ['airports-extended-clean.csv', 'schedule_airport.csv', '1Flight 1.xlsx', '1Flight 2.xlsx', '1Flight 3.xlsx', '1Flight 4.xlsx', '1Flight 5.xlsx', '1Flight 6.xlsx', '1Flight 7.xlsx', '30Flight 1.xlsx', '30Flight 2.xlsx', '30Flight 3.xlsx', '30Flight 4.xlsx', '30Flight 5.xlsx', '30Flight 6.xlsx', '30Flight 7.xlsx'])

try:
    if selected_file.endswith('.csv'):
        if selected_file == 'airports-extended-clean.csv':
            selected_df = pd.read_csv(selected_file, sep=';')
            selected_df = selected_df.drop_duplicates()
            selected_df = selected_df.dropna()
            selected_df['Continent'] = selected_df['Country'].map(country_to_continent)
            st.write(selected_df.head())  # Display the head of the DataFrame
    
            # Plot the airports longitudes by continent
            plot_mean_Latitude_by_continent(selected_df)
            plot_mean_Longitude_by_continent(selected_df)
            plot_mean_Altitude_by_continent(selected_df)

            
        elif selected_file == 'schedule_airport.csv':
            selected_df = pd.read_csv(selected_file)
            selected_df = selected_df.drop_duplicates()
            selected_df = selected_df.dropna()
            st.write(selected_df.head())  # Display the head of the DataFrame
            if 'STD' in selected_df.columns:
                selected_df['STD'] = pd.to_datetime(selected_df['STD'], format='%d/%m/%Y', dayfirst=True)
                plot_line_charts(selected_df)  # Plot the line chart
            else:
                st.warning("This file does not contain the 'STD' column.")
    else:
        selected_df = pd.read_excel(selected_file)
        selected_df = selected_df.drop_duplicates()
        selected_df = selected_df.dropna()
        st.write(selected_df.head())  # Display the head of the DataFrame
        plot_mean_values(selected_df)  # Plot the mean values


except Exception as e:
    st.error(f"An error occurred: {e}")
    
#fixing values
df1['Tz'] = df1['Tz'].replace('/N', 'Not assigned/Not assigned')

#splitting a column into multiple columns
split_result = df1['Tz'].str.split(pat='/', n=1, expand=True)
df1[['region', 'area']] = split_result
#sorting the dataframe to only contain airports
dfa = df1[df1['Type']== 'airport']

#colors variable
colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow']
#map creation
m = folium.Map(location=[0,0], zoom_start=2)

#creation of marker cluster
regiongr = {}
for region in dfa['region'].unique():
    regiongr[region]=MarkerCluster(name=region).add_to(m)

#creating the points on the map
for index, row in dfa.iterrows():
    lat = float(row['Latitude'].replace(',','.'))
    lng = float(row['Longitude'].replace(',','.'))
    color = colors[hash(row['region'])% len(colors)]
    folium.Marker([lat, lng], popup=row['Name'], icon=folium.Icon(color=color)).add_to(regiongr[row['region']])

#adding layer control
folium.LayerControl().add_to(m)

#making the map open in your browser
m.save('map.html')
webbrowser.open('map.html')

st.title("Airport Schedule Statistic Model")
st.write('''
         A statistical representation of the Airports Schedule 
         using a regression model, performance measures and 
         a scatterplot for clearer visualization
         ''')

over, prep, mod, perf = st.tabs(['Data Overview', 'Preprocessing', 'Logistic Regression Model', 'Performance Measures']) 

# Load data
def load_data():
    # Load your dataset here
    df = pd.read_csv('schedule_airport.csv')
    return df

# Main function
def main():
    with over:
        
        st.header('Overview of the "Airport Schedule"')
        df = pd.read_csv('schedule_airport.csv')
        flight = df.drop(columns = ['DL1', 'IX1', 'DL2', 'IX2'])
        
        # dropdown menus
        sort_by_flight = st.selectbox('Sort by flight number:', ['All'] + flight['FLT'].unique().tolist())
        sort_by_lsv = st.selectbox('Sort by inbound (L)/ outbound (S):', ['All'] + flight['LSV'].unique().tolist())
        sort_by_des = st.selectbox('Sort by destination:', ['All'] + flight['Org/Des'].unique().tolist())
        
        # filtering
        if sort_by_flight == 'All':
            sorted_df = flight
        else:
            sorted_df = flight[flight['FLT'] == sort_by_flight]

        if sort_by_lsv != 'All':
            sorted_df = sorted_df[sorted_df['LSV'] == sort_by_lsv]

        if sort_by_des != 'All':
            sorted_df = sorted_df[sorted_df['Org/Des'] == sort_by_des]

        # displaying
        st.write(sorted_df)
        
        st.header('Column name descriptions:')
        st.write('''
                 STD = date of departure
                 
                 FLT = flight number
                 
                 STA_STD_ltc = planned arrival in country of destination
                 
                 ATA_ATD_ltc = actual arrival in country of destination
                 
                 LSV = wether the flight is an inbound (L) flight or
                 an outbound (S) flight
                 
                 TAR = planned gate of arrival at destination airport
                 
                 GAT = actual gate of arrival at destination airport
                 
                 ACT = flight type
                 
                 RWY = landing strip
                 
                 RWC = time zone
                 
                 Identifier = the date, time and flight number respectively
                 
                 Org/Des = airport of destination
                 ''')
        
    with prep:
        st.header('Preprocessing step-by-step')
        st.write('''
                 Preprocessing is necesssary to ensure that your data is 
                 well-prepared for modeling, leading to better 
                 model performance and more reliable results
                 ''')
                 
        # Preprocess data
        def preprocess_data(data):
            # Perform necessary preprocessing
            df = pd.read_csv('schedule_airport.csv', nrows = 200)
            df.dropna(inplace = True)
            flight = df.drop(columns = ['DL1', 'IX1', 'DL2', 'IX2'])
            
            # Encode categorical variables using one-hot encoding
            data = pd.get_dummies(flight, columns = ['FLT', 'LSV', 'TAR', 'GAT', 'ACT', 'RWY', 'Org/Des'])

            # Define the target variable
            data['on_time_arrival'] = (data['ATA_ATD_ltc'] <= data['STA_STD_ltc'])

            # Split the data into features and target variable
            X = data.drop(columns = ['STD', 'STA_STD_ltc', 'ATA_ATD_ltc', 'RWC', 'Identifier', 'on_time_arrival'])
            y = data['on_time_arrival']

            return X, y
        
        # Train the logistic regression model
        def train_model(X_train, y_train):
            model = LogisticRegression()
            model.fit(X_train, y_train)
            return model
        
        st.write('''
                 Step 1.
                 During the preprocessing phase the data gets sampled, in this case
                 a sample size of nrows = 200 was used since the original data contained
                 300K+ data points
                 ''')
        st.write('''
                 Step 2.
                 The irrelevent columns and empty columns get dropped for 
                 a better overview of the data
                 ''')
        st.write('''
                 Step 3.
                 The categorical variables get the One-Hot Encoding technique,
                 to transform them into a format that can be provided 
                 for machine learning algorithms to improve model performance
                 ''')
        st.write('''
                 Step 4.
                 The target variable gets chosen, in this case this is
                 the time of arrival and finally the model gets trained
                 using logistic regression modeling
                 ''')
                 
        # Add a picture
        st.image("Preprocessing.png", caption = "Preprocessing code", use_column_width = True)

      
    with mod:
        st.header('Logistic Regression Model')
        st.write('''
                 Logistic regression is a powerful and versatile tool 
                 that is well-suited for many classification tasks, 
                 particularly when interpretability, simplicity, and 
                 computational efficiency are important considerations
                 ''')
        st.write('''
                 During logistic regression modeling the accuracy, precision,
                 recall, f1 and roc auc get calculated to be displayed later on
                 ''')
        st.header('Confusion Matrix using a Heatmap')
        st.write('''
                 For a visual representation a simple confusion matrix was also
                 encoded using sample data (n = 200) of the planned arrivals and 
                 actual arrivals
                 ''')
        
        # Function to calculate sensitivity
        def sensitivity(conf_matrix):
            TP = conf_matrix[1, 1]
            FN = conf_matrix[1, 0]
            return TP / (TP + FN)

        # Function to calculate specificity
        def specificity(conf_matrix):
            TN = conf_matrix[0, 0]
            FP = conf_matrix[0, 1]
            return TN / (TN + FP)

        # Function to calculate positive predictive value (PPV)
        def ppv(conf_matrix):
            TP = conf_matrix[1, 1]
            FP = conf_matrix[0, 1]
            return TP / (TP + FP)

        # Function to calculate negative predictive value (NPV)
        def npv(conf_matrix):
            TN = conf_matrix[0, 0]
            FN = conf_matrix[1, 0]
            return TN / (TN + FN)

        # Stratified Sampling
        stratified_sample = flight.groupby('LSV', group_keys = False).apply(lambda x: x.sample(min(len(x), 200)))

        # Combine Samples
        sample_flight = stratified_sample.sample(n = 200)
        sample_flight.head()

        # Compute confusion matrix
        conf_matrix = confusion_matrix(sample_flight['ATA_ATD_ltc'], sample_flight['STA_STD_ltc'])

        # Plot confusion matrix
        plt.figure(figsize = (8, 6))
        sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues', 
                    xticklabels = ['Not Arrived', 'Arrived'],
                    yticklabels = ['Not Arrived', 'Arrived'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(plt)

        # Calculate metrics
        sens = sensitivity(conf_matrix)
        spec = specificity(conf_matrix)
        ppv_score = ppv(conf_matrix)
        npv_score = npv(conf_matrix)

        # Display metrics
        st.write(f"Sensitivity: {sens}")
        st.write(f"Specificity: {spec}")
        st.write(f"Positive Predictive Value (PPV): {ppv_score}")
        st.write(f"Negative Predictive Value (NPV): {npv_score}")
        
        st.write('''
                 In conclusion, while a right-skewed heatmap in a confusion matrix,
                 the sensitivity, specificity, PPV and NPV
                 can provide insights into the model's performance, 
                 it is essential to interpret it within the context of 
                 the classification problem and consider other performance metrics 
                 to draw meaningful conclusions. For instance performance measures
                 given in the next tab
                 ''')
    
    with perf:
        st.header('Performance Measures')
        st.write('''
                 Performance measures are crucial for evaluating 
                 the effectiveness and reliability of a machine learning model
                 ''')
        
        def evaluate_model(model, X_test, y_test):
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)

            return accuracy, precision, recall, f1, roc_auc
        
        # Load the data
        df = load_data()

        # Preprocess the data
        X, y = preprocess_data(df)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the logistic regression model
        model = train_model(X_train, y_train)

        # Evaluate the model
        accuracy, precision, recall, f1, roc_auc = evaluate_model(model, X_test, y_test)

        # Display evaluation metrics
        st.write("Model Evaluation Metrics:")
        st.write(f"Accuracy: {accuracy}")
        st.write(f"Precision: {precision}")
        st.write(f"Recall: {recall}")
        st.write(f"F1 Score: {f1}")
        st.write(f"ROC AUC Score: {roc_auc}")
        
        st.write('''
                 In conclusion, these metrics provide insights into
                 different aspects of the model's performance, so
                 evaluating these metrics together can help in understanding
                 the model's strengths and weaknesses and making informed
                 decisions about its deployment
                 ''')
        
if __name__ == '__main__':
    main()        











