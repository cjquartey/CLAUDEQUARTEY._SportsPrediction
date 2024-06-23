import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

model = pickle.load(open('FINAL_XGBRegressor.pkl', 'rb'))
training_data = pd.read_csv('male_players (legacy).csv')
scaler = StandardScaler()
scaler.fit(training_data[['potential', 'value_eur', 'wage_eur', 'movement_reactions', 'mentality_composure']])

def calculate_percentile(prediction, training_data):
    min_rating = training_data['overall'].min()
    max_rating = training_data['overall'].max()
    
    center = (min_rating + max_rating) / 2
    distance_from_center = abs(prediction - center)
    max_distance = (max_rating - min_rating) / 2
    
    percentile = 100 * (1 - (distance_from_center / max_distance))
    return max(0, min(100, percentile))

def main():
    st.title("FIFA Player Rating Predictor")
    
    # Input fields for 5 features
    potential = st.number_input("Potential: ", min_value=0, max_value = 99, value=0)
    value_eur = st.number_input("Value (Euros): ", min_value=0, max_value = 1000000000, value=0)
    wage_eur = st.number_input("Wage (Euros): ", min_value=0, max_value = 1000000000, value=0)
    movement_reactions = st.number_input("Movement Reaction: ", min_value=0, max_value = 99, value=0)
    mentality_composure = st.number_input("Mentality Composure: ", min_value=0, max_value = 99, value=0)

    if st.button("Predict"):
        # Create a DataFrame with the input features
        df = pd.DataFrame({
            'potential': [potential],
            'value_eur': [value_eur],
            'wage_eur': [wage_eur],
            'movement_reactions': [movement_reactions],
            'mentality_composure': [mentality_composure]
        })
        
        # Transform the input data
        scaled_df = scaler.transform(df)
        
        # Predict using the model with scaled data
        prediction = model.predict(scaled_df)
        output = prediction[0]
        
        percentile = calculate_percentile(output, training_data)
        
        st.success(f'Predicted Player Rating: {output:.0f}')
        st.info(f'Percentile: {percentile:.2f}%')

if __name__ == "__main__":
    main()